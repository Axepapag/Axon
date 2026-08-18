from __future__ import annotations

import json
import random
from pathlib import Path

from substrate import assert_supported_text
from training.build_phase0b_exact_curriculum import (
    FAMILY_SPECS,
    build_curriculum,
    cap_emitted_source_groups,
    exact_legacy_target,
)
from training.build_phase1a_curriculum import build_source_chains, lossless_target_chunks
from training.trainer_slot import Phase0bCurriculum


def _write_inputs(root: Path) -> None:
    for input_family in FAMILY_SPECS.values():
        rows = []
        for index in range(40):
            rows.append(
                {
                    "family": input_family,
                    "source": f"{input_family}-source-{index}",
                    "active_field": {
                        "conversation_history": "history " + ("h" * 600),
                        "structured_knowledge": f"knowledge {index}",
                        "user_input": f"question {index}",
                    },
                    "target_delta": {
                        "region": "scratch" if input_family == "scratchpad_delta_v1" else "response_draft",
                        "op": "replace",
                        "text": f"exact legacy target {index}",
                    },
                }
            )
        (root / f"{input_family}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_phase0b_long_target_is_exactly_chained(tmp_path) -> None:
    path = tmp_path / "runtime_response_delta_v1.jsonl"
    target = " ".join(f"boundaryword{index:02d}" for index in range(12))
    row = {
        "family": "runtime_response_delta_v1",
        "source": "one-source",
        "active_field": {"conversation_history": "h" * 300, "user_input": "continue"},
        "target_delta": {"region": "response_draft", "op": "replace", "text": target},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    groups, stats = build_source_chains(
        path,
        "runtime_response_delta_v1",
        "runtime_response_delta_chain_v3",
        target_getter=exact_legacy_target,
        schema="axon_phase0b_delta_v3",
        prefer_boundaries=True,
    )
    chain = next(iter(groups.values()))
    assert stats["candidate_rows"] == 3
    assert "".join(item["target_delta"]["text"] for item in chain) == target
    assert all(len(item["target_delta"]["text"]) <= 64 for item in chain)
    assert all(item["budget_audit"]["silent_clips"] == 0 for item in chain)
    starts = [item["source"]["char_start"] for item in chain[1:]]
    assert all(not (target[start - 1].isalnum() and target[start].isalnum()) for start in starts)
    assert stats["boundary_cuts"] == 2


def test_lossless_chunker_hard_cuts_only_when_no_sensible_boundary() -> None:
    target = "a" * 70
    chunks = lossless_target_chunks(target, width=64, prefer_boundaries=True)
    assert "".join(chunk[2] for chunk in chunks) == target
    assert [chunk[3] for chunk in chunks] == ["hard", "final"]


def test_structured_rows_are_deleaked_and_deduplicated(tmp_path) -> None:
    path = tmp_path / "structured_knowledge_delta_v1.jsonl"
    rows = [
        {
            "source": f"source-{index}",
            "active_field": {
                "structured_knowledge": "word Jeffrey edge is a target person",
                "user_input": "Jeffrey is a",
            },
            "target_delta": {"region": "response_draft", "op": "replace", "text": "person"},
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    groups, stats = build_source_chains(
        path,
        "structured_knowledge_delta_v1",
        "structured_knowledge_delta_chain_v3",
        target_getter=exact_legacy_target,
        schema="axon_phase0b_delta_v3",
        prefer_boundaries=True,
        clean_visible_target=True,
        deduplicate_visible_target=True,
        emit_allowed_draft_modes=True,
    )
    emitted = [row for source_rows in groups.values() for row in source_rows]
    assert len(emitted) == 1
    assert "person" not in emitted[0]["active_field"]["conversation_history"].casefold()
    assert emitted[0]["quality_audit"]["target_leakage_replacements"] == 1
    assert "blank" not in emitted[0]["allowed_draft_modes"]
    assert stats["target_leakage_records"] == 2
    assert stats["duplicate_visible_target_records"] == 1


def test_source_balance_caps_records_without_truncating_selected_targets(tmp_path) -> None:
    path = tmp_path / "runtime_response_delta_v1.jsonl"
    rows = [
        {
            "source": "dominant-source",
            "active_field": {"conversation_history": "short history", "user_input": "respond"},
            "target_delta": {"region": "response_draft", "op": "replace", "text": f"answer {index}"},
        }
        for index in range(6)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    groups, stats = build_source_chains(
        path,
        "runtime_response_delta_v1",
        "runtime_response_delta_chain_v3",
        target_getter=exact_legacy_target,
        schema="axon_phase0b_delta_v3",
        max_records_per_source=2,
        emit_allowed_draft_modes=True,
    )
    selected = groups["dominant-source"]
    assert len(selected) == 2
    assert {row["source"]["source_balance_rank"] for row in selected} == {0, 1}
    assert all(row["target_delta"]["text"].startswith("answer ") for row in selected)
    assert all("blank" in row["allowed_draft_modes"] for row in selected)
    assert stats["candidate_records"] == 6
    assert stats["retained_records"] == 2
    assert stats["retained_sources"] == 1
    assert stats["source_balance_dropped_records"] == 4


def test_emitted_source_cap_excludes_whole_groups_without_truncation() -> None:
    over_cap = [{"chain": {"chain_id": "long-chain", "part_index": index}} for index in range(3)]
    within_cap = [{"chain": {"chain_id": "short-chain", "part_index": index}} for index in range(2)]
    eligible, audit = cap_emitted_source_groups(
        {"long-source": over_cap, "short-source": within_cap},
        max_examples=2,
    )
    assert eligible == {"short-source": within_cap}
    assert audit == {
        "emitted_source_cap": 2,
        "emitted_source_cap_excluded_groups": 1,
        "emitted_source_cap_excluded_rows": 3,
        "emitted_source_cap_eligible_groups": 1,
        "emitted_source_cap_eligible_rows": 2,
    }


def test_phase0b_exact_builder_and_loader_are_strict_and_split_by_source(tmp_path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    _write_inputs(source)
    manifest = build_curriculum(source, output, per_family=20, seed=19)

    assert manifest["schema"] == "axon_phase0b_exact_curriculum_manifest_v3"
    assert manifest["bootstrap_window"]["doctrine"] is False
    assert manifest["private_state_claim"] is False
    assert manifest["native_nonresponse_region_claim"] is False
    assert manifest["source_balance"]["max_emitted_examples_per_source"] == 24
    assert manifest["source_balance"]["truncate_chains"] is False
    sampler = manifest["training_sampler_contract"]
    assert sampler["required"] is True
    assert sampler["row_duplication_for_balance"] is False
    assert sampler["family_weights"] == {
        "runtime_response_delta_chain_v3": 0.30,
        "structured_knowledge_delta_chain_v3": 0.40,
        "scratchpad_delta_chain_v3": 0.30,
    }
    for family in FAMILY_SPECS:
        selected_audit = manifest["families"][family]["selected_audit"]
        assert selected_audit["rows"] == 20
        assert selected_audit["allowed_draft_modes"] == {"copy/partial": 20}
        assert selected_audit["broken_chains"] == 0
        assert selected_audit["avoidable_midword_boundaries"] == 0
        assert selected_audit["silent_clips"] == 0
        assert selected_audit["unsupported_substitutions"] == 0
        assert selected_audit["source_split_overlap"] == {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        }
        rows = [json.loads(line) for line in (output / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 20
        assert all(row["schema"] == "axon_phase0b_delta_v3" for row in rows)
        sources = {
            split: {row["source"]["record_id"] for row in rows if row["split"] == split}
            for split in ("train", "dev", "test")
        }
        assert sources["train"].isdisjoint(sources["dev"] | sources["test"])
        assert sources["dev"].isdisjoint(sources["test"])
        for row in rows:
            active = row["active_field"]
            answer = row["target_delta"]["text"]
            assert len(active["conversation_history"]) <= 256
            assert len(active["user_input"]) <= 64
            assert 0 < len(answer) <= 64
            assert row["budget_audit"]["silent_clips"] == 0
            assert row["budget_audit"]["unsupported_substitutions"] == 0
            assert row["allowed_draft_modes"] == ["copy", "partial"]
            assert_supported_text(active["conversation_history"] + active["user_input"] + answer)

    curriculum = Phase0bCurriculum(
        str(output),
        ",".join(FAMILY_SPECS),
        max_chars=64,
        history_chars=256,
        user_chars=64,
        rng=random.Random(19),
        max_examples=100,
    )
    assert all(row["mode"] == "phase0b_exact" and row["strict_budget"] for row in curriculum.examples)
    train_sources = {row["source"] for row in curriculum.examples}
    dev_sources = {row["source"] for row in curriculum.eval_cache}
    test_sources = {row["source"] for row in curriculum.test_cache}
    assert train_sources.isdisjoint(dev_sources | test_sources)
    assert dev_sources.isdisjoint(test_sources)
