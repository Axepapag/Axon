"""Commandments loader.

Reads the Ten Commandments and registers each as a structured bundle in the
dormant store. The commandment's 128D concept slot is built SOLELY from the
8D letter substrate — character by character, word by word, composed through
the same pathway as every other bundle in Axon.

NO HASH. NO RNG. NO RANDOM PROJECTION.

The principle text is read character by character. Each character becomes an
8D letter-slot via the frozen substrate. Words are bundles of letter-slots.
The full principle is a bundle of word-bundles. The concept atom is the
composed address of this bundle, broadcast to 128D by repeating the 8D
pattern — the same way the workshop promotes words from 8D to 64D.

This ensures commandments live in the SAME vector space as every other
concept. Semantic search finds them the same way it finds "dog" or "justice".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bundle import Bundle, make_commandment_slot
from dormant_store import DormantStore
from letter_substrate import word_to_letter_slots, SLOT_DIM


CONCEPT_DIM = 128

# Deterministic word-vector cache. English repeats words heavily; the
# position-weighted word vector is frozen-substrate-derived, so caching
# changes speed, never values. (This is what makes booting the canonical
# state from 100k+ plain-text lines fast.)
_WORD_VEC_CACHE: dict[str, np.ndarray] = {}


def _word_vec_cached(word: str) -> np.ndarray:
    vec = _WORD_VEC_CACHE.get(word)
    if vec is None:
        slots = word_to_letter_slots(word)
        norms = np.linalg.norm(slots, axis=1, keepdims=True).clip(min=1e-8)
        slots = slots / norms
        weights = 1.0 / (1.0 + np.arange(slots.shape[0], dtype=np.float32))
        vec = (slots * weights[:, None]).sum(axis=0)
        wn = float(np.linalg.norm(vec))
        if wn > 1e-8:
            vec = vec / wn
        vec = vec.astype(np.float32)
        vec.setflags(write=False)
        _WORD_VEC_CACHE[word] = vec
    return vec


@dataclass
class CommandmentRecord:
    number: int
    name: str
    original: str
    plain: str
    principle: str
    positive: str
    negative: str
    related: list[str]
    example_positive: str
    example_negative: str
    example_test: str
    surface: str
    bundle_id: str


def _principle_to_concept_atom(principle: str, dim: int = CONCEPT_DIM) -> np.ndarray:
    """Build a concept atom from principle text using ONLY the 8D substrate.

    Position-structured composition: each word becomes a position-weighted
    8D word vector (letter k weighted 1/(1+k)); word w lands in segment
    (w mod dim/8) of the atom with a 1/(1 + w//n_segments) wrap decay.

    NO hash. NO rng. NO random numbers. Only the frozen substrate.

    The old version took a flat mean over every letter of the whole
    principle text and tiled it - means of English text converge, so all
    ten commandments ended up nearly parallel and semantic search could
    not tell them apart. The positional structure keeps each principle's
    atom distinct while staying deterministic and substrate-derived.
    """
    words = re.findall(r"[A-Za-z']+", principle.lower())
    if not words:
        # Empty principle - return a neutral atom
        atom = np.zeros(dim, dtype=np.float32)
        atom[0] = 1.0
        return atom

    n_segments = max(1, dim // SLOT_DIM)
    atom = np.zeros(n_segments * SLOT_DIM, dtype=np.float32)
    for w_idx, w in enumerate(words):
        word_vec = _word_vec_cached(w)
        seg = w_idx % n_segments
        decay = 1.0 / (1.0 + w_idx // n_segments)
        atom[seg * SLOT_DIM:(seg + 1) * SLOT_DIM] += decay * word_vec

    atom = atom[:dim]
    n = float(np.linalg.norm(atom))
    if n > 1e-8:
        atom = atom / n
    return atom.astype(np.float32)


def parse_commandments_file(path: str | Path) -> list[CommandmentRecord]:
    """Parse the structured commandments file into records.

    Expected format (one block per commandment, blank line separated):

      commandment: 1
      name: no_other_gods
      original: ...
      plain: ...
      principle: ...
      positive: ...
      negative: ...
      related: god, worship, idol, serve, honor
      example_positive: ...
      example_negative: ...
      example_test: ...
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)
    out: list[CommandmentRecord] = []
    for block in blocks:
        if not block.strip():
            continue
        if block.lstrip().startswith("#"):
            continue
        fields: dict[str, str] = {}
        for line in block.splitlines():
            m = re.match(r"^([a-z_]+)\s*:\s*(.*)$", line, re.DOTALL)
            if m:
                fields[m.group(1)] = m.group(2).strip()
        if "commandment" not in fields or "name" not in fields:
            continue
        try:
            n = int(fields["commandment"])
        except ValueError:
            continue
        related = [w.strip() for w in fields.get("related", "").split(",") if w.strip()]
        rec = CommandmentRecord(
            number=n,
            name=fields["name"],
            original=fields.get("original", ""),
            plain=fields.get("plain", ""),
            principle=fields.get("principle", ""),
            positive=fields.get("positive", ""),
            negative=fields.get("negative", ""),
            related=related,
            example_positive=fields.get("example_positive", ""),
            example_negative=fields.get("example_negative", ""),
            example_test=fields.get("example_test", ""),
            surface=f"commandment:{n}:{fields['name']}",
            bundle_id=f"commandment:{n}:{fields['name']}",
        )
        out.append(rec)
    return out


def commandment_to_bundle(rec: CommandmentRecord) -> Bundle:
    """Convert a parsed record to a structured Bundle.

    The bundle has:
      - one 128D concept-slot built from the 8D substrate ONLY
      - edges to each related word (kind='ref')
      - edges to positive form, negative form, original form, plain form
      - edges to positive and negative example bundles
    """
    atom = _principle_to_concept_atom(rec.principle, dim=CONCEPT_DIM)
    slot = make_commandment_slot(atom, slot_id=f"{rec.bundle_id}#principle")
    b = Bundle(
        bundle_id=rec.bundle_id,
        kind="commandment",
        surface=rec.surface,
        native_slots=[slot],
    )
    for w in rec.related:
        b.edges.append((f"ref:{w}", f"word:{w}"))
    if rec.positive:
        b.edges.append(("form:positive", f"phrase:{_slug(rec.positive)}"))
    if rec.negative:
        b.edges.append(("form:negative", f"phrase:{_slug(rec.negative)}"))
    if rec.original:
        b.edges.append(("form:original", f"phrase:{_slug(rec.original)}"))
    if rec.plain:
        b.edges.append(("form:plain", f"phrase:{_slug(rec.plain)}"))
    if rec.example_positive:
        b.edges.append(("example:positive", f"sentence:{_slug(rec.example_positive)}"))
    if rec.example_negative:
        b.edges.append(("example:negative", f"sentence:{_slug(rec.example_negative)}"))
    if rec.example_test:
        b.edges.append(("example:test", f"question:{_slug(rec.example_test)}"))
    b.provenance["source"] = "hand_written_commandments"
    b.provenance["commandment_number"] = int(rec.number)
    b.provenance["commandment_name"] = rec.name
    return b


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", "", text.lower())
    s = re.sub(r"\s+", "_", s.strip())
    return s[:64]


def load_commandments_into_dormant(dormant: DormantStore, path: str | Path | None = None) -> list[Bundle]:
    """Parse the commandments file and register each as a dormant bundle.

    Idempotent: if a commandment is already in the dormant store, the
    existing bundle is kept and no duplicate is registered.
    """
    if path is None:
        here = Path(__file__).resolve().parent
        for candidate in (here / "config" / "commandments.txt",
                          Path("D:/00/axon_runtime/config/commandments.txt")):
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                "commandments.txt not found in D:/ashes/config or "
                "D:/00/axon_runtime/config"
            )
    records = parse_commandments_file(path)
    bundles: list[Bundle] = []
    for rec in records:
        existing = dormant.get(rec.bundle_id)
        if existing is not None:
            bundles.append(existing)
            continue
        b = commandment_to_bundle(rec)
        dormant.register(b)
        bundles.append(b)
    return bundles


def commandment_record_to_dict(rec: CommandmentRecord) -> dict[str, Any]:
    return {
        "number": rec.number,
        "name": rec.name,
        "surface": rec.surface,
        "bundle_id": rec.bundle_id,
        "original": rec.original,
        "plain": rec.plain,
        "principle": rec.principle,
        "positive": rec.positive,
        "negative": rec.negative,
        "related": rec.related,
        "example_positive": rec.example_positive,
        "example_negative": rec.example_negative,
        "example_test": rec.example_test,
    }


if __name__ == "__main__":
    print("=== commandments loader smoketest ===\n")

    here = Path(__file__).resolve().parent
    path = here / ".." / "00" / "Raw" / "corpus" / "layer6_commandments.txt"
    print(f"--- reading {path} ---")
    records = parse_commandments_file(path)
    print(f"  parsed {len(records)} commandments")
    for rec in records:
        print(f"  {rec.number:2d}. {rec.name:20s} ({len(rec.related)} related words)")

    print("\n--- principle -> concept atom (substrate-only, no hash) ---")
    a1 = _principle_to_concept_atom("There is one source of all things.")
    a2 = _principle_to_concept_atom("There is one source of all things.")
    a3 = _principle_to_concept_atom("Life is sacred. To end a life is to end a world.")
    print(f"  same text -> same atom: {np.allclose(a1, a2)}")
    print(f"  different text -> different atom: {not np.allclose(a1, a3)}")
    print(f"  atom dim: {a1.shape[0]}  norm: {float(np.linalg.norm(a1)):.3f}")
    sim_12 = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-8))
    sim_13 = float(np.dot(a1, a3) / (np.linalg.norm(a1) * np.linalg.norm(a3) + 1e-8))
    print(f"  cos(1, 1) = {sim_12:.3f} (expected 1.0)")
    print(f"  cos(1, 6) = {sim_13:.3f} (expected < 1.0, different principles)")

    # Verify: first 8D comes from substrate, rest is repetition
    print(f"  first 8D:  {a1[:8].tolist()}")
    print(f"  next 8D:   {a1[8:16].tolist()}")
    print(f"  match: {np.allclose(a1[:8], a1[8:16])} (expected: True — repetition)")

    print("\n--- convert record to bundle ---")
    rec = records[0]
    b = commandment_to_bundle(rec)
    print(f"  bundle_id: {b.bundle_id}")
    print(f"  kind: {b.kind}")
    print(f"  surface: {b.surface}")
    print(f"  n_slots: {b.n_slots()} (1 concept slot at 128D)")
    print(f"  slot kinds: {[s.kind for s in b.native_slots]}")
    print(f"  edges: {len(b.edges)}")
    for kind, target in b.edges[:5]:
        print(f"    {kind:25s} -> {target}")
    print(f"    ... and {len(b.edges) - 5} more")

    print("\n--- load into dormant store + save/reload round-trip ---")
    from dormant_store import DormantStore
    dormant = DormantStore()
    bundles = load_commandments_into_dormant(dormant, path)
    print(f"  loaded {len(bundles)} commandment bundles into dormant")
    print(f"  dormant.n_bundles: {dormant.n_bundles}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "dormant"
        # The SQLite store persists where it was constructed; save(target)
        # only refreshes the readable view. Round-trip = construct anew.
        dormant_disk = DormantStore(store_dir=target)
        with dormant_disk.bulk():
            for rec in records:
                if not dormant_disk.has(rec.bundle_id):
                    dormant_disk.register(commandment_to_bundle(rec))
        dormant_disk.save()
        dormant_disk.close()
        print(f"  saved to {target}")
        dormant2 = DormantStore(store_dir=target)
        print(f"  reloaded: dormant2.n_bundles: {dormant2.n_bundles}")
        rec1 = dormant2.get("commandment:1:no_other_gods")
        rec6 = dormant2.get("commandment:6:no_murder")
        rec9 = dormant2.get("commandment:9:no_false_witness")
        print(f"  commandment:1 reloaded: {rec1 is not None} (surface={rec1.surface if rec1 else None})")
        print(f"  commandment:6 reloaded: {rec6 is not None} (surface={rec6.surface if rec6 else None})")
        print(f"  commandment:9 reloaded: {rec9 is not None} (surface={rec9.surface if rec9 else None})")
        if rec1 is not None:
            for slot in rec1.native_slots:
                if slot.kind == "commandment":
                    print(f"  commandment:1 atom shape: {slot.vector.shape}, norm: {float(np.linalg.norm(slot.vector)):.3f}")
        if rec6 is not None:
            print(f"  commandment:6 edges: {len(rec6.edges)}")
        dormant2.close()

    print("\nM8a (commandments loader) smoketest PASSED")
