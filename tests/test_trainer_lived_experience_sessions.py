from __future__ import annotations

import hashlib
from pathlib import Path

from runtime.dormant import DormantExperienceStore, ExperienceRecord, RecoveredSourceSnapshot
from runtime.trainer import LivedExperienceSessionCompiler, TrainerControlPlane


def _record(source_hash: str, sequence: int, role: str, text: str) -> ExperienceRecord:
    return ExperienceRecord(
        record_kind="message",
        source_name="memory.db",
        source_sha256=source_hash,
        source_pointer=f"sqlite:memory.db:messages:{sequence}",
        sequence=sequence,
        exact_text=text,
        payload={"id": str(sequence), "role": role, "content": text},
    )


def test_trainer_compiles_exact_idempotent_sessions_without_copying_text(tmp_path: Path) -> None:
    state_root = tmp_path / "State"
    experience = DormantExperienceStore(state_root)
    snapshot = experience.root / "source_snapshots" / "source" / "memory.db"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"source")
    source_hash = hashlib.sha256(b"source").hexdigest()
    binding = RecoveredSourceSnapshot(
        source_name="memory.db",
        original_path="X:/protected/memory.db",
        size_bytes=6,
        sha256=source_hash,
        snapshot_relative_path="source_snapshots/source/memory.db",
    )
    records = (
        _record(source_hash, 0, "user", "First?\nEvery character."),
        _record(source_hash, 1, "assistant", "First answer!"),
        _record(source_hash, 2, "user", "Second—question"),
        _record(source_hash, 3, "assistant", "Second answer."),
    )
    imported = experience.publish_import(records, sources=(binding,), label="test-life")
    compiler = LivedExperienceSessionCompiler(experience)
    first = compiler.compile_observed_conversation((imported.import_id,))
    second = compiler.compile_observed_conversation((imported.import_id,))
    assert first.session_id == second.session_id
    assert len(first.examples) == 2
    assert not first.policy.serving_promotion_eligible
    assert first.examples[1].context_sequence_start == 0
    assert first.examples[1].context_sequence_end == 4
    serialized = str(first.to_canonical_dict())
    assert "First?" not in serialized
    assert records[0].record_id in serialized

    grounding = compiler.compile_heart_grounding((imported.import_id,))
    assert len(grounding.examples) == len(records)
    with TrainerControlPlane.active(state_root=state_root) as control:
        path = control.publish_session(first)
        assert path.exists()
        assert control.publish_session(first) == path
