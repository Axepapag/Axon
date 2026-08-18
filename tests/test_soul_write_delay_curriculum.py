from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CONTEXT_END,
    CONTEXT_START,
    CORE_WRITABLE_REGIONS,
    N_SLOTS,
    PROPOSAL_END,
    PROPOSAL_START,
    USER_END,
    USER_START,
    PhysicalRole,
)
from training.build_multitick_curriculum import (
    SourceSpec,
    canonical_json_bytes,
    decode_lossless_escapes,
)
from training.build_soul_write_delay_curriculum import (
    COUNTERFACTUAL_CONDITIONS,
    PHYSICAL_PROFILE,
    WRITABLE_REGIONS,
    CausalLeakageError,
    CurriculumDiversityError,
    build_soul_write_delay_curriculum,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _read_episodes(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _prepared_row(index: int, *, lineage: str | None = None) -> dict:
    return {
        "family": "soul_write_delay_source",
        "lineage_id": lineage or f"lineage-{index}",
        "fact_text": f"Code {index} is value {100 + index}.",
        "evidence_text": f"Ledger {index} records value {100 + index}.",
        "recall_query": f"Recall item for cue {index}.",
        "distractors": [f"Unrelated delay task {index}."],
    }


def test_causal_contract_determinism_donors_hashes_and_source_immutability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.jsonl"
    _write_jsonl(source, [_prepared_row(index) for index in range(12)])
    before_bytes = source.read_bytes()
    before_hash = _sha256(source)
    before_mtime = source.stat().st_mtime_ns
    output = tmp_path / "out"

    manifest = build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source)],
        output,
        seed=17,
    )
    first_bytes = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    repeated = build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source)],
        output,
        seed=17,
    )
    second_bytes = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }

    assert manifest == repeated
    assert first_bytes == second_bytes
    assert source.read_bytes() == before_bytes
    assert _sha256(source) == before_hash
    assert source.stat().st_mtime_ns == before_mtime
    assert manifest["sources"][0]["sha256_before"] == before_hash
    assert manifest["sources"][0]["sha256_after"] == before_hash
    assert manifest["sources"][0]["read_only_verified"] is True
    assert manifest["split_leakage_audit"]["passed"] is True
    assert manifest["donor_identity_audit"]["passed"] is True
    assert manifest["split_counts"] == {"train": 8, "dev": 2, "test": 2}

    episodes = _read_episodes(output)
    by_id = {episode["episode_id"]: episode for episode in episodes}
    lineage_splits: dict[str, set[str]] = {}
    for episode in episodes:
        lineage_splits.setdefault(episode["source_lineage"], set()).add(
            episode["split"]
        )
        assert episode["runtime_field_contract"]["total_slots"] == N_SLOTS
        assert episode["runtime_field_contract"]["writable_regions"] == [
            "response_draft",
            "scratch",
        ]
        tick_a = episode["tick_a"]
        fact = tick_a["soul_write_target"]["target_soul_trace"]
        visible_a = tick_a["active_view"]["visible_region_texts"][
            "structured_knowledge"
        ]
        assert fact in visible_a
        assert "Evidence:" in visible_a
        credit = tick_a["soul_write_target"]["write_credit"]
        assert credit["immediate_write_weight"] == 1.0
        assert credit["delayed_recall_weight"] == 1.0
        assert credit["credit_assignment_tick_id"] == episode["tick_b"][
            "tick_id"
        ]

        assert len(episode["delay_ticks"]) >= 1
        assert all(
            delay["answer_leakage_audit"]["passed"]
            for delay in episode["delay_ticks"]
        )
        tick_b = episode["tick_b"]
        assert tick_b["answer_leakage_audit"]["passed"] is True
        assert tick_b["recall_supervision"]["requires_soul_state"] is True
        assert tick_b["recall_supervision"]["field_answer_available"] is False
        assert tick_b["recall_supervision"]["target_region"] == "response_draft"
        for region_text in tick_b["active_view"][
            "visible_region_texts"
        ].values():
            assert fact.casefold() not in region_text.casefold()

        counterfactuals = episode["counterfactuals"]
        ids = [
            counterfactuals[condition]["condition_id"]
            for condition in COUNTERFACTUAL_CONDITIONS
        ]
        assert len(set(ids)) == 4
        separation = counterfactuals["identity_separation"]
        assert all(
            separation[field] is True
            for field in (
                "donor_episode_differs",
                "donor_lineage_differs",
                "donor_cluster_differs",
                "donor_fact_differs",
                "donor_same_split",
                "condition_ids_unique",
            )
        )
        donor = by_id[counterfactuals["swapped"]["donor_episode_id"]]
        assert donor["split"] == episode["split"]
        assert donor["source_lineage"] != episode["source_lineage"]
        assert (
            donor["lineage_cluster_id"] != episode["lineage_cluster_id"]
        )
    assert all(len(splits) == 1 for splits in lineage_splits.values())

    for file_info in manifest["output_files"].values():
        assert _sha256(output / file_info["path"]) == file_info["sha256"]
    payload = copy.deepcopy(manifest)
    payload_hash = payload.pop("manifest_payload_sha256")
    assert _sha256_bytes(canonical_json_bytes(payload)) == payload_hash
    sidecar_hash = (output / "manifest.sha256").read_text(
        encoding="ascii"
    ).split()[0]
    assert sidecar_hash == _sha256(output / "manifest.json")


def test_exact_v4_adapter_derives_fact_evidence_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "exact-v4.jsonl"
    rows = []
    for index in range(4):
        rows.append(
            {
                "schema": "axon_multitick_exact_v4",
                "episode_id": f"upstream-{index}",
                "source_lineage": f"upstream-lineage-{index}",
                "initial_field": {
                    "structured_knowledge": {
                        "text": f"Recovered record {index}.",
                        "spans": [],
                    }
                },
                "ticks": [
                    {
                        "target_region": "response_draft",
                        "complete_proposed_region_text": (
                            f"Recovered answer {index}."
                        ),
                    }
                ],
            }
        )
    _write_jsonl(source, rows)
    output = tmp_path / "out"

    build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source, adapter="exact_v4")],
        output,
        seed=3,
    )
    episodes = _read_episodes(output)

    assert len(episodes) == 4
    for episode in episodes:
        assert (
            episode["provenance"]["upstream_schema"]
            == "axon_multitick_exact_v4"
        )
        assert episode["provenance"]["upstream_episode_id"].startswith(
            "upstream-"
        )
        assert episode["provenance"]["fact_origin"].startswith("ticks")
        assert "Recovered record" in episode["tick_a"]["active_field"][
            "structured_knowledge"
        ]
        assert episode["tick_b"]["answer_leakage_audit"]["passed"] is True


def test_unicode_escape_is_lossless_and_diary_is_default_excluded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unicode-private.jsonl"
    private_text = "private continuity note"
    rows = []
    for index in range(4):
        rows.append(
            {
                "lineage_id": f"unicode-{index}",
                "fact_text": f"Signal \U0001f525 {index}.",
                "evidence_text": f"Witness said \u201chot\u201d {index}.",
                "diary_text": private_text,
            }
        )
    _write_jsonl(source, rows)
    output = tmp_path / "out"

    manifest = build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source)],
        output,
        unsupported_policy="escape",
    )
    episodes = _read_episodes(output)

    assert manifest["diary"]["included"] is False
    for episode in episodes:
        encoded = episode["tick_b"]["recall_supervision"][
            "complete_expected_answer"
        ]
        decoded = decode_lossless_escapes(encoded)
        assert "\U0001f525" in decoded
        assert episode["privacy"]["diary_source_present"] is True
        assert episode["privacy"]["diary_excluded"] is True
        assert episode["privacy"]["diary_included"] is False
        assert any(
            audit["policy"] == "lossless_codepoint_escape"
            for audit in episode["provenance"]["text_transform_audits"]
        )
    derived_text = "\n".join(
        path.read_text(
            encoding="ascii" if path.suffix == ".sha256" else "utf-8"
        )
        for path in output.iterdir()
        if path.is_file()
    )
    assert private_text not in derived_text
    assert "\U0001f525" not in derived_text


def test_recall_query_leak_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "leaking.jsonl"
    fact = "The hidden answer is cedar."
    rows = [
        {
            "lineage_id": "leak-a",
            "fact_text": fact,
            "evidence_text": "The ledger records cedar.",
            "recall_query": f"Repeat this answer: {fact}",
        },
        _prepared_row(2),
    ]
    _write_jsonl(source, rows)
    before = source.read_bytes()

    with pytest.raises(CausalLeakageError):
        build_soul_write_delay_curriculum(
            [SourceSpec.jsonl(source)],
            tmp_path / "out",
        )
    assert source.read_bytes() == before


def test_tag_overwide_source_fact_is_excluded_without_clipping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "overwide.jsonl"
    # The physical proposal window is 64 slots, but the runtime-rendered
    # ``[response_draft]\n`` tag leaves only 47 payload slots.
    overwide_fact = "x" * 48
    rows = [
        {
            "lineage_id": "overwide",
            "fact_text": overwide_fact,
            "evidence_text": "short evidence",
        },
        *[_prepared_row(index) for index in range(4)],
    ]
    _write_jsonl(source, rows)
    output = tmp_path / "out"

    manifest = build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source)],
        output,
    )
    derived = "\n".join(
        path.read_text(
            encoding="ascii" if path.suffix == ".sha256" else "utf-8"
        )
        for path in output.iterdir()
        if path.is_file()
    )

    assert manifest["episodes"] == 4
    assert manifest["skipped"]["field_view_ineligible"] == 1
    assert overwide_fact not in derived


def test_production_gate_rejects_generic_fact_concentration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generic.jsonl"
    rows = [
        {
            "lineage_id": f"generic-{index}",
            "fact_text": "Repeated generic safety rule.",
            "evidence_text": "Repeated generic evidence.",
        }
        for index in range(300)
    ]
    _write_jsonl(source, rows)

    with pytest.raises(CurriculumDiversityError, match="diversity gate"):
        build_soul_write_delay_curriculum(
            [SourceSpec.jsonl(source)],
            tmp_path / "out",
            require_production_gate=True,
        )
    assert not (tmp_path / "out").exists()


def test_transitive_near_and_exact_lineages_share_clusters_and_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clusters.jsonl"
    rows = [
        {
            "lineage_id": "near-a",
            "fact_text": "Alpha One!",
            "evidence_text": "Note A!",
        },
        {
            "lineage_id": "near-bridge",
            "fact_text": "alpha one",
            "evidence_text": "note a",
        },
        {
            "lineage_id": "near-bridge",
            "fact_text": "Beta Two!",
            "evidence_text": "Note B!",
        },
        {
            "lineage_id": "near-c",
            "fact_text": "beta two",
            "evidence_text": "note b",
        },
        {
            "lineage_id": "exact-a",
            "fact_text": "Exact fact 77.",
            "evidence_text": "Exact evidence 77.",
        },
        {
            "lineage_id": "exact-b",
            "fact_text": "Exact fact 77.",
            "evidence_text": "Exact evidence 77.",
        },
    ]
    rows.extend(_prepared_row(index + 20) for index in range(8))
    _write_jsonl(source, rows)
    output = tmp_path / "out"

    manifest = build_soul_write_delay_curriculum(
        [SourceSpec.jsonl(source)],
        output,
        seed=9,
    )
    episodes = _read_episodes(output)

    def selected(*names: str) -> list[dict]:
        return [
            episode
            for episode in episodes
            if any(
                episode["source_lineage"].endswith(f":{name}")
                for name in names
            )
        ]

    transitive = selected("near-a", "near-bridge", "near-c")
    assert len(transitive) == 4
    assert len({episode["lineage_cluster_id"] for episode in transitive}) == 1
    assert len({episode["split"] for episode in transitive}) == 1

    exact = selected("exact-a", "exact-b")
    assert len(exact) == 2
    assert len({episode["lineage_cluster_id"] for episode in exact}) == 1
    assert len({episode["split"] for episode in exact}) == 1
    assert manifest["split_leakage_audit"]["passed"] is True
    assert manifest["donor_identity_audit"]["passed"] is True


def test_runtime_field_constants_are_reused_exactly() -> None:
    assert WRITABLE_REGIONS == frozenset(
        region.value for region in CORE_WRITABLE_REGIONS
    )
    assert WRITABLE_REGIONS == frozenset({"scratch", "response_draft"})
    assert PHYSICAL_PROFILE == (
        (
            "context_projection",
            CONTEXT_START,
            CONTEXT_END,
            int(PhysicalRole.CONTEXT),
        ),
        ("user_input", USER_START, USER_END, int(PhysicalRole.USER)),
        ("proposal", PROPOSAL_START, PROPOSAL_END, int(PhysicalRole.PROPOSAL)),
    )
    assert (CONTEXT_START, CONTEXT_END) == (0, 256)
    assert (USER_START, USER_END) == (256, 320)
    assert (PROPOSAL_START, PROPOSAL_END) == (320, N_SLOTS)
    assert [region.value for region in CANONICAL_REGION_ORDER] == [
        "conversation_history",
        "user_input",
        "structured_knowledge",
        "situation_awareness",
        "tool_results",
        "advisor_input",
        "task_state",
        "scratch",
        "response_draft",
        "diary",
    ]
