from __future__ import annotations

import hashlib
import json

import pytest

from runtime.dormant import DormantExperienceStore, ExperienceRecord
from runtime.field import canonical_sha256
from training.first_form_curriculum import (
    FirstFormCurriculumCompiler,
    load_first_form_curriculum,
    publish_first_form_curriculum,
)


def _source_for_split(split: str) -> str:
    for index in range(10_000):
        source = f"synthetic-{split}-{index}"
        lineage = canonical_sha256(
            {
                "schema": "axon-ffcs-source-window-v1",
                "source_name": source,
                "window": 0,
            }
        )
        bucket = int(hashlib.sha256(lineage.encode()).hexdigest()[:8], 16) % 10
        observed = "train" if bucket < 8 else "heldout" if bucket == 8 else "regression"
        if observed == split:
            return source
    raise AssertionError(f"failed to construct a {split} lineage")


def _experience_store(state_root) -> DormantExperienceStore:
    store = DormantExperienceStore(state_root)
    records = []
    sources = []
    for split in ("train", "heldout", "regression"):
        source_name = _source_for_split(split)
        source = store.publish_inline_source_snapshot(
            source_name=source_name,
            original_path=f"memory://{source_name}",
            exact_bytes=f"exact source for {split}".encode(),
        )
        sources.append(source)
        for sequence, (kind, role, text) in enumerate(
            (
                ("message", "user", f"question-{split}"),
                ("message", "assistant", f"answer-{split}-λ"),
                ("diary", "", f"memory-{split}-🧠"),
            )
        ):
            records.append(
                ExperienceRecord(
                    record_kind=kind,
                    source_name=source_name,
                    source_sha256=source.sha256,
                    source_pointer=f"record:{sequence}",
                    sequence=sequence,
                    exact_text=text,
                    payload={"role": role},
                    occurred_at=f"2026-08-20T12:00:0{sequence}-05:00",
                )
            )
    store.publish_import(records, sources=sources, label="ffcs-test")
    return store


def test_ffcs_compile_publish_load_and_exact_split_lineage(tmp_path) -> None:
    state_root = tmp_path / "State"
    curriculum = FirstFormCurriculumCompiler(_experience_store(state_root)).compile_abc(
        identity_text="Axon is Axon.\nEvery brother attends canonical Identity.",
        requested_counts=(("A", (1, 1, 1)), ("B", (1, 1, 1)), ("C", (1, 1, 1))),
    )

    assert curriculum.actual_family_split_counts == curriculum.requested_family_split_counts
    assert len(curriculum.cases) == 9
    for family in ("B", "C"):
        rows = [item for item in curriculum.cases if item.family == family]
        by_lineage: dict[str, set[str]] = {}
        for row in rows:
            by_lineage.setdefault(row.lineage_id, set()).add(row.episode.split)
        assert all(len(splits) == 1 for splits in by_lineage.values())
        assert all(row.source_record_ids for row in rows)

    path = publish_first_form_curriculum(curriculum, state_root=state_root)
    restored = load_first_form_curriculum(path)
    assert restored.manifest_id == curriculum.manifest_id
    assert restored.to_canonical_dict() == curriculum.to_canonical_dict()
    assert publish_first_form_curriculum(restored, state_root=state_root) == path


def test_ffcs_load_rejects_tampered_target_identity(tmp_path) -> None:
    state_root = tmp_path / "State"
    curriculum = FirstFormCurriculumCompiler(_experience_store(state_root)).compile_abc(
        identity_text="Axon is Axon.",
        requested_counts=(("A", (1, 1, 1)), ("B", (1, 1, 1)), ("C", (1, 1, 1))),
    )
    path = publish_first_form_curriculum(curriculum, state_root=state_root)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["cases"][0]["episode"]["targets"][0]["target_id"] = "0" * 64
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match="target identity mismatch"):
        load_first_form_curriculum(path)
