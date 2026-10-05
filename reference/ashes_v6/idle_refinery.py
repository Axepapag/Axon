"""Idle refinery: drains the dump bucket into structured knowledge and the training bucket.

The idle refinery runs on idle ticks (no pending user input, draft is
either empty or stable). It does three jobs:

  1. Drain `State/Dormant/Dump/` — raw tick captures — into structured
     knowledge and episodic memory in active. Each dump file's user_text
     and draft_text are ingested; the resulting word bundles are added
     to the active state's structured_knowledge (for semantic search
     reuse) and to the training bucket (for offline training).

  2. Promote vocabulary that has been referenced but not yet promoted.
     (The workshop does this on lookup, so this is mostly defensive
     for items in the registry that are still thin.)

  3. Build a high-level summary of what's in the training bucket so
     the runtime can decide if it has material to train on.

The refinery is **idempotent**: each dump file is processed exactly
once. After processing, the dump file is moved to
`State/Dormant/Dump/processed/`.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from bundle import Bundle
from training_bucket import TrainingExample, TrainingBucket
from active_state import segment_into_words


class IdleRefinery:
    """Drains the dump bucket and feeds the training bucket."""

    def __init__(self, state_dir: str | Path,
                 training_bucket: TrainingBucket,
                 active_state: Any,
                 workshop: Any,
                 dormant: Any,
                 archive_dir: str | Path | None = None):
        self.state_dir = Path(state_dir)
        self.training_bucket = training_bucket
        self.active = active_state
        self.workshop = workshop
        self.dormant = dormant
        self.dump_dir = self.state_dir / "Dormant" / "Dump"
        self.processed_dir = self.dump_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None

    def drain_dump_bucket(self, max_items: int = 16) -> dict[str, int]:
        """Process up to `max_items` dump files. Returns counts.

        Each dump file is a JSONL/JSON capture from a past tick. We
        extract the user_text and draft_text, segment into words,
        build bundles via the workshop, and push to the training
        bucket. After processing, the file is moved to `processed/`.

        Returns a stats dict: {"processed", "skipped", "examples_pushed",
                               "bundles_added_to_structured"}.
        """
        stats = {"processed": 0, "skipped": 0, "examples_pushed": 0,
                 "bundles_added_to_structured": 0}
        if not self.dump_dir.exists():
            return stats
        # List unprocessed dump captures, oldest first
        paths = sorted(
            [p for p in self.dump_dir.glob("*.json*") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
        )[:max_items]
        for path in paths:
            ok, n_pushed, n_struct = self._process_dump_file(path)
            if ok:
                stats["processed"] += 1
                stats["examples_pushed"] += n_pushed
                stats["bundles_added_to_structured"] += n_struct
                # Move to processed
                try:
                    target = self.processed_dir / path.name
                    shutil.move(str(path), str(target))
                except Exception:
                    pass
            else:
                stats["skipped"] += 1
        return stats

    def _process_dump_file(self, path: Path) -> tuple[bool, int, int]:
        """Process one dump file. Returns (ok, n_examples_pushed, n_structured)."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return False, 0, 0
        if not text:
            return False, 0, 0
        # The dump file might be JSON or JSONL. We try the JSON path first.
        import json
        data = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                break
            except Exception:
                continue
        if data is None:
            try:
                data = json.loads(text)
            except Exception:
                return False, 0, 0
        # Extract text. The dump_capture schema has tick_id, user_text,
        # draft_text, active_summary, roles.
        user_text = data.get("user_text", "") or ""
        draft_text = data.get("draft_text", "") or ""
        if not user_text and not draft_text:
            return False, 0, 0
        n_pushed = 0
        n_struct = 0
        for text_segment, source, speaker in [
            (user_text, "user_input", "user"),
            (draft_text, "draft_text", "axon"),
        ]:
            if not text_segment:
                continue
            words = segment_into_words(text_segment)
            if len(words) < 2:
                continue
            # Build word bundles and pull their concept atoms
            bundles: list[Bundle] = []
            for w in words:
                b = self.workshop.lookup_or_create(w)
                bundles.append(b)
            # Pull the word-atom concept dim (could vary if graduated)
            atom_dim = None
            for b in bundles:
                cs = b.concept_slot()
                if cs is not None:
                    atom_dim = cs.native_dim
                    break
            if atom_dim is None:
                # All bundles are thin (no concept slot yet). Skip.
                continue
            # Build (n_words, atom_dim) array. Pad/truncate if needed.
            atoms = np.zeros((len(bundles), atom_dim), dtype=np.float32)
            for i, b in enumerate(bundles):
                cs = b.concept_slot()
                if cs is not None and cs.native_dim == atom_dim:
                    atoms[i] = cs.vector
            # Build the example
            example = TrainingExample(
                surfaces=words,
                word_atoms=atoms,
                speaker=speaker,
                source=source,
                created_tick=self.active.n_ticks_processed,
            )
            ok, _ = example.safety_pass()
            if not ok:
                continue
            self.training_bucket.push(example)
            n_pushed += 1
            # Also add the bundles to structured knowledge (for semantic
            # search reuse in future ticks)
            for b in bundles:
                if b.bundle_id not in {x.bundle_id for x in self.active.structured_knowledge}:
                    self.active.structured_knowledge.append(b)
                    n_struct += 1
        return True, n_pushed, n_struct

    def promote_pending_vocabulary(self) -> int:
        """Promote any thin bundles in the registry that are ready."""
        promoted = self.workshop.promote_pending()
        # Register any newly promoted bundles in dormant.
        for b in promoted:
            if not self.dormant.has(b.bundle_id):
                self.dormant.register(b)
        return len(promoted)

    def tick_idle(self, max_items: int = 4) -> dict[str, Any]:
        """One idle pass. Returns the combined stats."""
        drain_stats = self.drain_dump_bucket(max_items=max_items)
        n_promoted = self.promote_pending_vocabulary()
        return {
            **drain_stats,
            "vocab_promotions": n_promoted,
            "bucket_size": len(self.training_bucket),
        }


if __name__ == "__main__":
    print("=== idle refinery smoketest ===\n")

    import json
    import tempfile
    from renderer import BundleRegistry
    from workshop import Workshop
    from active_state import ActiveState
    from dormant_store import DormantStore

    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        archive_dir = Path(tmp) / "archive"
        (state_dir / "Dormant" / "Dump").mkdir(parents=True)
        bucket = TrainingBucket(bucket_dir=state_dir / "Training" / "bucket")
        registry = BundleRegistry()
        workshop = Workshop(registry, promote_threshold=1, atom_dim=64)
        dormant = DormantStore(store_dir=state_dir / "Dormant")
        active = ActiveState(registry=registry, workshop=workshop,
                              state_dir=state_dir / "Active")
        # Write a fake dump file
        dump = state_dir / "Dormant" / "Dump" / "tick_00000001.json"
        dump.write_text(json.dumps({
            "tick_id": "tick_00000001",
            "user_text": "the dog runs fast",
            "draft_text": "the dog is fast",
        }), encoding="utf-8")
        # Run the refinery
        refinery = IdleRefinery(state_dir, bucket, active, workshop, dormant)
        stats = refinery.tick_idle(max_items=4)
        print(f"refinery stats: {stats}")
        assert stats["processed"] == 1
        assert stats["examples_pushed"] == 2  # user_text + draft_text
        assert stats["vocab_promotions"] >= 0
        # Verify the dump file moved
        assert not dump.exists()
        assert (state_dir / "Dormant" / "Dump" / "processed" / "tick_00000001.json").exists()

    print("\nM-refinery (idle refinery) smoketest PASSED")
