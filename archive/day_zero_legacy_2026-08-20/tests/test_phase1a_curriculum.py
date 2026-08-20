from __future__ import annotations

import json
import random
from pathlib import Path

from substrate import assert_supported_text
from training.build_phase1a_curriculum import FAMILY_SPECS, build_curriculum
from training.trainer_slot import Phase0bCurriculum


def _write_inputs(root: Path) -> None:
    for input_family in FAMILY_SPECS.values():
        rows = []
        for index in range(36):
            row = {
                "family": input_family,
                "source": f"{input_family}-source-{index}",
                "active_field": {
                    "conversation_history": "h" * (260 + index),
                    "structured_knowledge": f"knowledge {index}",
                    "user_input": f"question {index}",
                },
                "target_delta": {"region": "response_draft", "op": "replace", "text": f"answer {index}"},
            }
            if input_family == "scratchpad_delta_v1":
                row["secondary_delta"] = {"region": "response_draft", "op": "replace", "text": f"step {index}"}
            rows.append(row)
        (root / f"{input_family}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_phase1a_builder_is_exact_bounded_and_source_disjoint(tmp_path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    _write_inputs(source)
    manifest = build_curriculum(source, output, per_family=20, seed=7)

    assert manifest["bootstrap_window"]["doctrine"] is False
    assert manifest["private_state_claim"] is False
    for family in FAMILY_SPECS:
        rows = [json.loads(line) for line in (output / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 20
        assert {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")} == {
            "train": 16,
            "dev": 2,
            "test": 2,
        }
        sources = {
            split: {row["source"]["record_id"] for row in rows if row["split"] == split}
            for split in ("train", "dev", "test")
        }
        assert sources["train"].isdisjoint(sources["dev"] | sources["test"])
        assert sources["dev"].isdisjoint(sources["test"])
        for row in rows:
            active = row["active_field"]
            target = row["target_delta"]["text"]
            assert len(active["conversation_history"]) <= 256
            assert len(active["user_input"]) <= 64
            assert len(target) <= 64
            assert row["budget_audit"]["silent_clips"] == 0
            assert row["budget_audit"]["unsupported_substitutions"] == 0
            assert_supported_text(active["conversation_history"] + active["user_input"] + target)


def test_phase1a_loader_keeps_train_and_eval_sources_separate(tmp_path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    _write_inputs(source)
    build_curriculum(source, output, per_family=20, seed=11)

    curriculum = Phase0bCurriculum(
        str(output),
        ",".join(FAMILY_SPECS),
        max_chars=64,
        history_chars=256,
        user_chars=64,
        rng=random.Random(11),
        max_examples=100,
    )
    assert all(row["mode"] == "phase1a" and row["split"] == "train" for row in curriculum.examples)
    assert all(row["mode"] == "phase1a" and row["split"] == "dev" for row in curriculum.eval_cache)
    assert all(row["mode"] == "phase1a" and row["split"] == "test" for row in curriculum.test_cache)
    train_sources = {row["source"] for row in curriculum.examples}
    dev_sources = {row["source"] for row in curriculum.eval_cache}
    test_sources = {row["source"] for row in curriculum.test_cache}
    assert train_sources.isdisjoint(dev_sources | test_sources)
    assert dev_sources.isdisjoint(test_sources)
    all_rows = curriculum.examples + curriculum.eval_cache + curriculum.test_cache
    assert all(len(row["conversation_history"]) <= 256 for row in all_rows)
