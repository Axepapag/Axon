"""Training bucket: a queue of validated training examples for the offline core.

The bucket is populated by:
  - The IdleRefinery (drains `State/Dormant/Dump/` into structured knowledge
    and episodic memory; the episodic memory is fed into the bucket).
  - Vocabulary promotions logged in the workshop.
  - User input episodes (after they graduate from `conversation_history`).

The bucket is FIFO. When the offline trainer pops an item, it:
  1. Verifies the item passes the safety pass.
  2. Trains the offline core for one step on the item.
  3. Archives the item off-site (`D:/AxonArchive/training/`) with a
     metadata sidecar.
  4. Removes the item from the bucket.

No item is ever trained on twice by the same core. The archive is the
*only* place the example exists after consumption; the runtime never
reads the archive, the external trainer (in parallel) may.

**Storage: per-example .npz files, no JSON envelope.**

    State/Training/bucket/<example_id>.npz

Each example holds:
  - word_atoms: (K, D) array, the concept-dim atoms of the words.
  - surfaces: object array of length K, the surface forms.
  - speaker: object scalar, who produced the example.
  - source: object scalar, what produced it ('user_input', 'dump_refine', 'promotion').
  - created_tick: int, when it entered the bucket.
  - n_words: int, the K.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _safe_filename(example_id: str) -> str:
    return example_id.replace(":", "_").replace("/", "_") + ".npz"


class TrainingExample:
    """One validated training example for the offline core.

    Holds a sequence of word-atom concept vectors and the surfaces
    that produced them. The shape is (n_words, atom_dim) where atom_dim
    is the per-example dim (v6.4c allows 16D, 32D, 64D, 128D, …).
    """

    def __init__(self, surfaces: list[str], word_atoms: np.ndarray,
                 speaker: str = "user", source: str = "user_input",
                 created_tick: int = 0, example_id: str | None = None):
        if word_atoms.ndim != 2:
            raise ValueError(f"word_atoms must be 2D, got shape {word_atoms.shape}")
        if word_atoms.shape[0] != len(surfaces):
            raise ValueError(
                f"word_atoms rows {word_atoms.shape[0]} != surfaces {len(surfaces)}"
            )
        self.example_id = example_id or f"ex:{uuid.uuid4().hex[:12]}"
        self.surfaces = list(surfaces)
        self.word_atoms = word_atoms.astype(np.float32)
        self.speaker = str(speaker)
        self.source = str(source)
        self.created_tick = int(created_tick)
        self.created_at = time.time()
        self.atom_dim = int(word_atoms.shape[1])

    def safety_pass(self) -> tuple[bool, str]:
        """Validate the example before training. Returns (ok, reason)."""
        if self.word_atoms.shape[0] < 2:
            return False, "fewer than 2 words"
        if self.word_atoms.shape[0] > 10000:
            return False, "too many words (corpus line too long)"
        if not np.isfinite(self.word_atoms).all():
            return False, "non-finite values in word_atoms"
        if not all(s.strip() for s in self.surfaces):
            return False, "empty surface form"
        return True, "ok"


class TrainingBucket:
    """FIFO queue of training examples. npz storage. No size cap.

    The bucket is the on-board trainer's only source of training data.
    Items are added by the IdleRefinery and other producers; items are
    popped by the OfflineTrainer. Popped items are archived off-site
    and removed from disk.
    """

    def __init__(self, bucket_dir: str | Path | None = None):
        self.bucket_dir = Path(bucket_dir) if bucket_dir is not None else None
        # In-memory queue of example_ids, in FIFO order. The actual data
        # lives in <bucket_dir>/<safe(example_id)>.npz.
        self._queue: deque[str] = deque()
        # Cache: example_id -> TrainingExample
        self._cache: dict[str, TrainingExample] = {}
        if self.bucket_dir is not None:
            self._ensure_dir()
            self.load()

    def _ensure_dir(self) -> None:
        if self.bucket_dir is not None:
            self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._queue)

    def n_total(self) -> int:
        """Total items in the queue (queued + cached but not yet popped)."""
        return len(self._queue) + len(self._cache)

    def push(self, example: TrainingExample) -> None:
        """Add an example to the back of the queue."""
        if self.bucket_dir is not None:
            self._ensure_dir()
            path = self.bucket_dir / _safe_filename(example.example_id)
            np.savez_compressed(
                path,
                example_id=np.array(example.example_id, dtype=object),
                surfaces=np.array(example.surfaces, dtype=object),
                word_atoms=example.word_atoms,
                speaker=np.array(example.speaker, dtype=object),
                source=np.array(example.source, dtype=object),
                created_tick=np.array(example.created_tick),
                created_at=np.array(example.created_at, dtype=np.float64),
                atom_dim=np.array(example.atom_dim),
            )
        self._queue.append(example.example_id)
        self._cache[example.example_id] = example

    def peek(self) -> TrainingExample | None:
        """Return the next example without popping it."""
        if not self._queue:
            return None
        eid = self._queue[0]
        if eid in self._cache:
            return self._cache[eid]
        # Fall through to disk
        ex = self._load_from_disk(eid)
        if ex is not None:
            self._cache[eid] = ex
        return ex

    def pop(self) -> TrainingExample | None:
        """Pop the next example from the front. Caller is responsible for archiving."""
        if not self._queue:
            return None
        eid = self._queue.popleft()
        ex = self._cache.pop(eid, None)
        if ex is None:
            ex = self._load_from_disk(eid)
        # Remove from disk
        if self.bucket_dir is not None:
            path = self.bucket_dir / _safe_filename(eid)
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
        return ex

    def discard(self, example_id: str) -> None:
        """Remove an example from the queue by id (used by the safety pass on failure)."""
        if example_id in self._cache:
            del self._cache[example_id]
        if self.bucket_dir is not None:
            path = self.bucket_dir / _safe_filename(example_id)
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
        # Filter the queue (deque doesn't support remove-by-value efficiently
        # for large queues; for our scale this is fine).
        try:
            self._queue.remove(example_id)
        except ValueError:
            pass

    def _load_from_disk(self, example_id: str) -> TrainingExample | None:
        if self.bucket_dir is None:
            return None
        path = self.bucket_dir / _safe_filename(example_id)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            return TrainingExample(
                surfaces=[str(s) for s in data["surfaces"]],
                word_atoms=data["word_atoms"],
                speaker=str(data["speaker"]),
                source=str(data["source"]),
                created_tick=int(data["created_tick"]),
                example_id=str(data["example_id"]),
            )
        except Exception:
            return None

    def load(self) -> None:
        """Load the bucket from disk. Items in disk order (FIFO)."""
        if self.bucket_dir is None or not self.bucket_dir.exists():
            return
        # Sort by mtime (FIFO by arrival)
        paths = sorted(self.bucket_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for path in paths:
            try:
                data = np.load(path, allow_pickle=True)
                eid = str(data["example_id"])
                self._queue.append(eid)
            except Exception:
                continue

    def clear(self) -> None:
        """Drop everything from the bucket (used for testing)."""
        self._queue.clear()
        self._cache.clear()
        if self.bucket_dir is not None and self.bucket_dir.exists():
            for path in self.bucket_dir.glob("*.npz"):
                try:
                    path.unlink()
                except Exception:
                    pass


class Archive:
    """One-way write-only archive of consumed training examples.

    The runtime writes to <archive_dir>/<timestamp>_<core_id>_<hash>.npz
    and never reads. The external trainer (in parallel) may read this.
    """

    def __init__(self, archive_dir: str | Path | None = None):
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        if self.archive_dir is not None:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive(self, example: TrainingExample, core_id: str,
                tick: int, loss: float) -> Path | None:
        """Archive a consumed training example. Returns the archive path."""
        if self.archive_dir is None:
            return None
        ts = time.strftime("%Y%m%dT%H%M%S")
        # Hash the example's word atoms to get a stable suffix.
        h = int(np.sum(example.word_atoms) * 1e6) & 0xFFFFFFFF
        safe_core = core_id.replace(":", "_").replace("/", "_")
        path = self.archive_dir / f"{ts}_{safe_core}_t{tick:08d}_{h:08x}.npz"
        # Write the example plus a metadata sidecar
        try:
            np.savez_compressed(
                path,
                example_id=np.array(example.example_id, dtype=object),
                surfaces=np.array(example.surfaces, dtype=object),
                word_atoms=example.word_atoms,
                speaker=np.array(example.speaker, dtype=object),
                source=np.array(example.source, dtype=object),
                created_tick=np.array(example.created_tick),
                atom_dim=np.array(example.atom_dim),
                consumed_tick=np.array(tick),
                consumed_core=np.array(core_id, dtype=object),
                consumed_loss=np.array(loss, dtype=np.float64),
                archived_at=np.array(time.time(), dtype=np.float64),
            )
        except Exception:
            return None
        return path


if __name__ == "__main__":
    print("=== training bucket smoketest ===\n")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bucket_dir = Path(tmp) / "bucket"
        archive_dir = Path(tmp) / "archive"
        bucket = TrainingBucket(bucket_dir=bucket_dir)
        archive = Archive(archive_dir=archive_dir)

        # Push a few examples
        for i, (text, source) in enumerate([
            (["the", "dog", "runs"], "user_input"),
            (["a", "cat", "sleeps"], "dump_refine"),
            (["hello", "world"], "promotion"),
        ]):
            atoms = np.random.randn(len(text), 64).astype(np.float32)
            atoms = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
            ex = TrainingExample(text, atoms, source=source, created_tick=i)
            bucket.push(ex)
        print(f"after 3 pushes: queue={len(bucket)}")

        # Peek + safety pass
        peeked = bucket.peek()
        ok, reason = peeked.safety_pass()
        print(f"peek: id={peeked.example_id} safety={ok} reason={reason}")

        # Pop and archive
        popped = bucket.pop()
        loss = 0.42
        path = archive.archive(popped, "core_128A", tick=10, loss=loss)
        print(f"popped: {popped.example_id}, archived to: {path.name}")
        print(f"after pop: queue={len(bucket)}")
        print(f"archive contents: {len(list(archive_dir.glob('*.npz')))} files")

        # Reload
        bucket2 = TrainingBucket(bucket_dir=bucket_dir)
        print(f"reloaded: queue={len(bucket2)}")

    print("\nM-bucket (training bucket) smoketest PASSED")
