"""Dormant store v3: the state lives FULLY IN RAM; the disk is plain text.

Jeff's law for persistence: no shady files. Everything in State/ is a
normal .txt file a human can open, read, and edit. The folder mirrors
the canonical state in RAM both ways:

  * anything that changes in RAM is written back to the folder,
  * anything that changes in the folder is loaded back into RAM
    (sync_from_disk(), called by the runtime every tick).

THE RENDERER IS THE BRIDGE. Disk holds English; RAM holds bundles.
Going out, a bundle is serialized as its surface text plus its edges and
provenance. Coming in, the frozen substrate re-encodes the text into
letter slots and the deterministic birth-atom composition rebuilds the
concept atom. Atoms that are NOT derivable from their text (e.g. the
commandments, whose atoms come from their principle text) are written
into the line as plain decimal numbers - still text, still readable.

Layout (one file per bundle kind):

    State/Dormant/
        words.txt  concepts.txt  episodes.txt  commandments.txt  ...

Line format (one bundle per line):

    <id> | <kind> | slots: <letters|atom|letters+atom> <D>D | <surface>
        [| edges: kind -> target; ...] [| prov: {json}] [| atom: [..]]

**No size cap.** The store grows as bundles are promoted. RAM is the
only limit, per the source of truth.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from bundle import (
    Bundle, Slot, VALID_KINDS, SLOT_KIND_LETTER, SLOT_KIND_WORD,
    SLOT_KIND_CONCEPT, SLOT_KIND_COMMANDMENT, make_letter_slot,
)
from letter_substrate import SLOT_DIM


def _birth_word_atom(surface: str, dim: int) -> np.ndarray:
    from workshop import _initial_word_atom
    return _initial_word_atom(surface, dim)


def _birth_concept_atom(text: str, dim: int) -> np.ndarray:
    from commandments import _principle_to_concept_atom
    return _principle_to_concept_atom(text, dim)


def _birth_atom(kind: str, surface: str, dim: int) -> np.ndarray:
    if kind == "word":
        return _birth_word_atom(surface, dim)
    return _birth_concept_atom(surface, dim)


_ATOM_SLOT_KIND = {
    "word": SLOT_KIND_WORD,
    "commandment": SLOT_KIND_COMMANDMENT,
}


class _Record:
    """The in-RAM canonical form of one dormant bundle."""
    __slots__ = ("kind", "surface", "has_letters", "atom", "edges",
                 "provenance", "created_at")

    def __init__(self, kind: str, surface: str, has_letters: bool,
                 atom: np.ndarray | None, edges: list[tuple[str, str]],
                 provenance: dict, created_at: float = 0.0):
        self.kind = kind
        self.surface = surface
        self.has_letters = has_letters
        self.atom = atom
        self.edges = edges
        self.provenance = provenance
        self.created_at = created_at


def _escape(text: str) -> str:
    return str(text).replace("|", "¦").replace("\n", " ").replace("\r", " ")


def _record_to_line(bundle_id: str, r: _Record) -> str:
    spec = ("letters+atom" if (r.has_letters and r.atom is not None)
            else "letters" if r.has_letters else "atom")
    dim = r.atom.shape[0] if r.atom is not None else SLOT_DIM
    parts = [bundle_id, r.kind, f"slots: {spec} {dim}D", _escape(r.surface)]
    if r.edges:
        parts.append("edges: " + "; ".join(
            f"{_escape(k)} -> {_escape(t)}" for k, t in r.edges))
    if r.provenance:
        parts.append("prov: " + json.dumps(r.provenance, ensure_ascii=False,
                                           sort_keys=True))
    if r.atom is not None:
        birth = _birth_atom(r.kind, r.surface, int(dim))
        cos = float(np.dot(r.atom, birth) /
                    (np.linalg.norm(r.atom) * np.linalg.norm(birth) + 1e-12))
        if cos < 0.999:
            # Not derivable from the surface: write the numbers as text.
            nums = ", ".join(f"{x:.4f}" for x in r.atom)
            parts.append(f"atom: [{nums}]")
    return " | ".join(parts)


_SLOTS_RE = re.compile(r"^slots:\s*(letters\+atom|letters|atom)\s+(\d+)D$")


def _line_to_record(line: str) -> tuple[str, _Record] | None:
    fields = [f.strip() for f in line.split(" | ")]
    if len(fields) < 4:
        return None
    bundle_id, kind = fields[0], fields[1]
    if kind not in VALID_KINDS:
        return None
    m = _SLOTS_RE.match(fields[2])
    if not m:
        return None
    spec, dim = m.group(1), int(m.group(2))
    surface = fields[3].replace("¦", "|")
    edges: list[tuple[str, str]] = []
    provenance: dict = {}
    atom: np.ndarray | None = None
    for extra in fields[4:]:
        if extra.startswith("edges:"):
            for pair in extra[len("edges:"):].split(";"):
                if "->" in pair:
                    k, _, t = pair.partition("->")
                    edges.append((k.strip().replace("¦", "|"),
                                  t.strip().replace("¦", "|")))
        elif extra.startswith("prov:"):
            try:
                provenance = json.loads(extra[len("prov:"):].strip())
            except Exception:
                provenance = {"unparsed": extra[len("prov:"):].strip()[:200]}
        elif extra.startswith("atom:"):
            try:
                nums = extra[len("atom:"):].strip().strip("[]")
                atom = np.array([float(x) for x in nums.split(",")],
                                dtype=np.float32)
            except Exception:
                atom = None
    has_letters = spec in ("letters", "letters+atom")
    if spec != "letters" and atom is None:
        atom = _birth_atom(kind, surface, dim)
    return bundle_id, _Record(kind, surface, has_letters, atom, edges,
                              provenance)


class DormantStore:
    """RAM-canonical dormant store mirrored to plain-text files."""

    def __init__(self, store_dir: str | Path | None = None):
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self._records: dict[str, _Record] = {}
        self._index_cache: dict[int, tuple[np.ndarray, list[str]]] = {}
        self._index_stale = True
        self._dirty_kinds: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        self._dirty = False
        if self.store_dir is not None:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._migrate_sqlite_if_any()
            self._load_all_files()

    # ------------------------------------------------------------------
    # Core dict-like API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, bundle_id: str) -> bool:
        return bundle_id in self._records

    def __iter__(self) -> Iterator[Bundle]:
        for bid in list(self._records.keys()):
            b = self.get(bid)
            if b is not None:
                yield b

    @property
    def n_bundles(self) -> int:
        return len(self._records)

    @property
    def bundle_ids(self) -> list[str]:
        return list(self._records.keys())

    def has(self, bundle_id: str) -> bool:
        return bundle_id in self._records

    @property
    def n_thin(self) -> int:
        return sum(1 for r in self._records.values() if not r.edges)

    @property
    def n_mature(self) -> int:
        return sum(1 for r in self._records.values() if r.edges)

    def thin_bundle_ids(self) -> list[str]:
        return [bid for bid, r in self._records.items() if not r.edges]

    def mature_bundle_ids(self) -> list[str]:
        return [bid for bid, r in self._records.items() if r.edges]

    def get(self, bundle_id: str) -> Bundle | None:
        """Materialize the bundle from RAM. Letter slots are rebuilt from
        the surface through the frozen substrate (deterministic)."""
        r = self._records.get(bundle_id)
        if r is None:
            return None
        slots: list[Slot] = []
        if r.has_letters:
            slots.extend(make_letter_slot(c) for c in r.surface)
        if r.atom is not None:
            skind = _ATOM_SLOT_KIND.get(r.kind, SLOT_KIND_CONCEPT if r.kind != "word"
                                        else SLOT_KIND_WORD)
            slots.append(Slot(slot_id=f"{bundle_id}#atom", kind=skind,
                              vector=r.atom.copy(), role="address"))
        b = Bundle(bundle_id=bundle_id, kind=r.kind, surface=r.surface,
                   decode=r.surface, native_slots=slots,
                   edges=list(r.edges), provenance=dict(r.provenance),
                   created_at=r.created_at)
        b.freeze()
        return b

    def get_by_surface(self, surface: str) -> Bundle | None:
        for bid, r in self._records.items():
            if r.surface == surface:
                return self.get(bid)
        return None

    def get_atom(self, bundle_id: str, kind: str = "") -> np.ndarray | None:
        r = self._records.get(bundle_id)
        return r.atom if r is not None else None

    # ------------------------------------------------------------------
    # Registration (RAM first; text mirror flushed on save)
    # ------------------------------------------------------------------

    @contextmanager
    def bulk(self):
        """Compatibility: batch imports. Mirrors flush once at the end."""
        try:
            yield self
        finally:
            self.save()

    def _record_from_bundle(self, bundle: Bundle) -> _Record:
        has_letters = any(s.kind == SLOT_KIND_LETTER for s in bundle.native_slots)
        atom = None
        best_dim = -1
        for s in bundle.native_slots:
            if s.kind != SLOT_KIND_LETTER and s.native_dim > best_dim:
                atom = s.vector.astype(np.float32)
                best_dim = s.native_dim
        return _Record(bundle.kind, bundle.surface or bundle.decode or bundle.bundle_id,
                       has_letters, atom, list(bundle.edges),
                       dict(bundle.provenance),
                       bundle.created_at or time.time())

    def register(self, bundle: Bundle, freeze: bool = True) -> None:
        if bundle.bundle_id in self._records:
            raise ValueError(f"bundle {bundle.bundle_id} already exists in store")
        if freeze:
            bundle.freeze()
        self._records[bundle.bundle_id] = self._record_from_bundle(bundle)
        self._touch(bundle.kind)

    def update(self, bundle: Bundle) -> None:
        if bundle.bundle_id not in self._records:
            raise ValueError(f"bundle {bundle.bundle_id} not in store")
        bundle.freeze()
        self._records[bundle.bundle_id] = self._record_from_bundle(bundle)
        self._touch(bundle.kind)

    def add_edge(self, bundle_id: str, kind: str, target: str) -> None:
        r = self._records.get(bundle_id)
        if r is None:
            raise ValueError(f"bundle {bundle_id} not in store")
        r.edges.append((kind, target))
        self._touch(r.kind)

    def _touch(self, kind: str) -> None:
        self._dirty_kinds.add(kind)
        self._index_stale = True
        self._dirty = True

    # ------------------------------------------------------------------
    # Semantic search (per-dimension sub-indexes, Section 9.2)
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        by_dim: dict[int, tuple[list[np.ndarray], list[str]]] = {}
        for bid, r in self._records.items():
            if r.atom is None:
                continue
            d = int(r.atom.shape[0])
            vecs, ids = by_dim.setdefault(d, ([], []))
            vecs.append(r.atom)
            ids.append(bid)
        self._index_cache = {}
        for d, (vecs, ids) in by_dim.items():
            M = np.stack(vecs, axis=0)
            norms = np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-8)
            self._index_cache[d] = (M / norms, ids)
        self._index_stale = False

    def search_atom(self, query: np.ndarray, top_k: int = 1,
                    min_similarity: float = 0.0) -> list[tuple[str, float]]:
        if query.ndim != 1:
            raise ValueError(f"query must be 1D, got shape {query.shape}")
        if self._index_stale:
            self._rebuild_index()
        entry = self._index_cache.get(int(query.shape[0]))
        if entry is None:
            return []
        M, ids = entry
        qn = float(np.linalg.norm(query))
        if qn < 1e-8:
            return []
        sims = M @ (query.astype(np.float32) / qn)
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(ids[i], float(sims[i])) for i in top if sims[i] >= min_similarity]

    def address_matrix(self) -> tuple[np.ndarray, list[str]]:
        ids, vecs = [], []
        for bid, r in self._records.items():
            if r.kind == "word" and r.atom is not None:
                ids.append(bid)
                vecs.append(r.atom)
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32), []
        max_d = max(v.shape[0] for v in vecs)
        out = np.zeros((len(vecs), max_d), dtype=np.float32)
        for i, v in enumerate(vecs):
            out[i, : v.shape[0]] = v
        return out, ids

    def edges_of(self, bundle_id: str) -> list[tuple[str, str]]:
        r = self._records.get(bundle_id)
        return list(r.edges) if r is not None else []

    # ------------------------------------------------------------------
    # The text mirror: RAM -> disk
    # ------------------------------------------------------------------

    def _kind_path(self, kind: str) -> Path:
        return self.store_dir / f"{kind}s.txt"

    def save(self, store_dir: str | Path | None = None) -> None:
        """Flush dirty kinds to their .txt files."""
        if store_dir is not None:
            self.store_dir = Path(store_dir)
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._dirty_kinds = {r.kind for r in self._records.values()}
        if self.store_dir is None:
            self._dirty = False
            return
        for kind in sorted(self._dirty_kinds):
            lines = [
                _record_to_line(bid, r)
                for bid, r in sorted(self._records.items())
                if r.kind == kind
            ]
            path = self._kind_path(kind)
            if lines:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self._file_mtimes[path.name] = path.stat().st_mtime
            elif path.exists():
                path.unlink()
                self._file_mtimes.pop(path.name, None)
        self._dirty_kinds.clear()
        self._dirty = False

    def close(self) -> None:
        self.save()

    # ------------------------------------------------------------------
    # The text mirror: disk -> RAM
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> dict[str, _Record]:
        out: dict[str, _Record] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _line_to_record(line)
            if parsed is not None:
                out[parsed[0]] = parsed[1]
        return out

    def _load_all_files(self) -> None:
        for path in sorted(self.store_dir.glob("*.txt")):
            if path.name.endswith("_VIEW.txt"):
                continue
            kind = path.stem[:-1] if path.stem.endswith("s") else path.stem
            if kind not in VALID_KINDS:
                continue
            self._records.update(self._parse_file(path))
            self._file_mtimes[path.name] = path.stat().st_mtime
        self._index_stale = True

    def sync_from_disk(self) -> int:
        """If Jeff edited a mirror file, fold the edits into RAM.
        Returns the number of bundles added/changed/removed."""
        if self.store_dir is None:
            return 0
        n_changed = 0
        for path in sorted(self.store_dir.glob("*.txt")):
            if path.name.endswith("_VIEW.txt"):
                continue
            kind = path.stem[:-1] if path.stem.endswith("s") else path.stem
            if kind not in VALID_KINDS:
                continue
            mtime = path.stat().st_mtime
            if self._file_mtimes.get(path.name) == mtime:
                continue
            on_disk = self._parse_file(path)
            in_ram = {bid for bid, r in self._records.items() if r.kind == kind}
            # Removed lines -> removed bundles.
            for bid in in_ram - set(on_disk):
                del self._records[bid]
                n_changed += 1
            # New or edited lines -> upsert.
            for bid, rec in on_disk.items():
                old = self._records.get(bid)
                if old is None or _record_to_line(bid, old) != _record_to_line(bid, rec):
                    self._records[bid] = rec
                    n_changed += 1
            self._file_mtimes[path.name] = mtime
        if n_changed:
            self._index_stale = True
        return n_changed

    # ------------------------------------------------------------------
    # One-time migration from the SQLite era
    # ------------------------------------------------------------------

    def _migrate_sqlite_if_any(self) -> None:
        db_path = self.store_dir / "dormant.sqlite"
        if not db_path.exists():
            return
        import sqlite3
        print(f"[DORMANT] migrating {db_path} to plain-text mirror...", flush=True)
        t0 = time.time()
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        edge_map: dict[str, list[tuple[str, str]]] = {}
        for bid, k, t in cur.execute(
                "SELECT bundle_id, kind, target FROM edges ORDER BY bundle_id, idx"):
            edge_map.setdefault(bid, []).append((k, t))
        atom_map: dict[str, np.ndarray] = {}
        letters: set[str] = set()
        for bid, skind, dim, blob in cur.execute(
                "SELECT bundle_id, kind, dim, vector FROM slots"):
            if skind == SLOT_KIND_LETTER:
                letters.add(bid)
            else:
                atom_map[bid] = np.frombuffer(blob, dtype=np.float32)[:dim].copy()
        n = 0
        for bid, kind, surface, decode, created_at, prov in cur.execute(
                "SELECT bundle_id, kind, surface, decode, created_at,"
                " provenance FROM bundles"):
            self._records[bid] = _Record(
                kind, surface or decode or bid, bid in letters,
                atom_map.get(bid), edge_map.get(bid, []),
                json.loads(prov or "{}"), created_at or 0.0)
            n += 1
        con.close()
        self._dirty_kinds = {r.kind for r in self._records.values()}
        self.save()
        db_path.rename(db_path.with_suffix(".sqlite.migrated"))
        print(f"[DORMANT] migrated {n:,} bundles to .txt in "
              f"{time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # Human-readable summary (the mirror itself is the full view now)
    # ------------------------------------------------------------------

    def render_view(self, per_kind: int = 60) -> str:
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("AXON DORMANT STORE (the library - full text in State/Dormant/*.txt)")
        lines.append(f"bundles: {self.n_bundles:,}   thin: {self.n_thin:,}   "
                     f"mature: {self.n_mature:,}   "
                     f"written: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 72)
        by_kind: dict[str, list[tuple[str, int]]] = {}
        for bid, r in self._records.items():
            by_kind.setdefault(r.kind, []).append((r.surface or bid, len(r.edges)))
        for kind in sorted(by_kind):
            group = sorted(by_kind[kind])
            lines.append("")
            lines.append(f"-- {kind.upper()} ({len(group):,}) -> {kind}s.txt "
                         + "-" * max(1, 44 - len(kind)))
            for surface, n_edges in group[:per_kind]:
                text = surface if len(surface) <= 64 else surface[:61] + "..."
                lines.append(f"  {text:<66s} edges: {n_edges}")
            if len(group) > per_kind:
                lines.append(f"  ... and {len(group) - per_kind:,} more in {kind}s.txt")
        lines.append("")
        return "\n".join(lines)

    def load(self, store_dir: str | Path | None = None) -> None:
        """Compatibility shim."""
        if store_dir is not None and Path(store_dir) != self.store_dir:
            self.__init__(store_dir)
        else:
            self.sync_from_disk()


if __name__ == "__main__":
    print("=== dormant store (text-canonical) smoketest ===\n")

    import tempfile
    from bundle import build_letter_word_bundle, build_word_bundle_with_atom

    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "dormant"
        store = DormantStore(store_dir=store_dir)

        a = build_word_bundle_with_atom("alpha", _birth_word_atom("alpha", 64))
        b = build_word_bundle_with_atom("beta", _birth_word_atom("beta", 64))
        c = build_letter_word_bundle("gamma")
        store.register(a)
        store.register(b)
        store.register(c)
        store.add_edge("word:beta", "is_a", "word:alpha")
        store.save()
        print(f"registered 3, saved. words.txt:")
        for line in (store_dir / "words.txt").read_text(encoding="utf-8").splitlines():
            print(f"  {line[:110]}")

        # Round trip
        store2 = DormantStore(store_dir=store_dir)
        assert store2.n_bundles == 3
        beta = store2.get("word:beta")
        assert beta.edges == [("is_a", "word:alpha")]
        assert beta.frozen
        atom = store2.get_atom("word:alpha")
        hits = store2.search_atom(atom, top_k=2)
        assert hits[0][0] == "word:alpha"
        print(f"\nround-trip OK; search: {hits}")

        # Disk -> RAM: Jeff edits the file by hand
        time.sleep(0.05)
        words = (store_dir / "words.txt").read_text(encoding="utf-8")
        words += "word:delta | word | slots: letters+atom 64D | delta | prov: {\"source\": \"hand_edit\"}\n"
        (store_dir / "words.txt").write_text(words, encoding="utf-8")
        changed = store2.sync_from_disk()
        print(f"hand-edit sync: {changed} change(s); has delta: {store2.has('word:delta')}")
        assert store2.has("word:delta")
        d = store2.get("word:delta")
        assert d.surface == "delta" and d.concept_slot() is not None

        # RAM -> disk
        store2.add_edge("word:delta", "synonym", "word:alpha")
        store2.save()
        assert "synonym -> word:alpha" in (store_dir / "words.txt").read_text(encoding="utf-8")
        print("RAM-edit flushed back to words.txt OK")

    print("\nM5 (dormant store, text-canonical) smoketest PASSED")
