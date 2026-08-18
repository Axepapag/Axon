from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
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
    READ_PAGING_PROTOCOL,
    USER_END,
    USER_START,
    PhysicalRole,
    proposal_payload_capacity,
    proposal_tail_offset,
)
from substrate import assert_supported_text
from training.build_multitick_curriculum import (
    CANONICAL_REGIONS,
    FAMILY_ADVISOR,
    FAMILY_DIARY,
    FAMILY_GRAMMAR,
    FAMILY_NOOP,
    FAMILY_SCRATCH,
    FAMILY_STRUCTURED,
    FAMILY_TASK,
    FAMILY_TOOL,
    ExactTextEncoder,
    PHYSICAL_384_PROFILE,
    REGION_TARGET_BUDGET,
    SEGMENT_COUNT,
    WRITABLE_REGIONS,
    SealedTargetError,
    SourceSpec,
    apply_segment_delta,
    audit_cross_split_leakage,
    build_curriculum,
    decode_lossless_escapes,
    field_sha256,
    sha256_text,
    sqlite_ro_uri,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_episodes(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _family_row(family: str, lineage: str, suffix: str) -> dict:
    common = {
        "family": family,
        "lineage_id": lineage,
        "conversation_history": f"history {lineage} {suffix}",
        "user_input": f"request {lineage} {suffix}",
        "initial_draft": f"draft {lineage} {suffix}",
        "situation_awareness": f"working on {lineage} {suffix}",
    }
    if family == FAMILY_GRAMMAR:
        common["surface"] = f"teh answer {lineage} {suffix}"
        common["target"] = f"the answer {lineage} {suffix}"
    elif family == FAMILY_STRUCTURED:
        common["structured_knowledge"] = f"fact {lineage} {suffix}"
        common["response_after"] = f"grounded answer {lineage} {suffix}"
    elif family == FAMILY_TOOL:
        common["tool_results"] = f"tool result {lineage} {suffix}"
        common["response_after"] = f"integrated answer {lineage} {suffix}"
    elif family == FAMILY_SCRATCH:
        common["scratch_plan"] = f"plan {lineage} {suffix}"
        common["response_after"] = f"planned answer {lineage} {suffix}"
    elif family == FAMILY_ADVISOR:
        common["advisor_input"] = f"advisor evidence {lineage} {suffix}"
        common["response_after"] = f"revised answer {lineage} {suffix}"
    elif family == FAMILY_TASK:
        common["task_state"] = f"active task {lineage} {suffix}"
        common["scratch_plan"] = f"preserve and resume {lineage} {suffix}"
        common["task_after"] = f"resume task {lineage} {suffix}"
        common["response_after"] = f"resumed answer {lineage} {suffix}"
    elif family == FAMILY_NOOP:
        pass
    else:
        raise AssertionError(f"unexpected family {family}")
    return common


def test_build_is_deterministic_source_disjoint_and_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    families = [
        FAMILY_GRAMMAR,
        FAMILY_STRUCTURED,
        FAMILY_TOOL,
        FAMILY_SCRATCH,
        FAMILY_ADVISOR,
        FAMILY_TASK,
        FAMILY_NOOP,
    ]
    rows: list[dict] = []
    # Two records per lineage prove that split assignment happens before rows.
    for index in range(10):
        family = families[index % len(families)]
        rows.append(_family_row(family, f"group{index}", "a"))
        rows.append(_family_row(family, f"group{index}", "b"))
    _write_jsonl(source, rows)
    source_before = source.read_bytes()
    source_hash = _sha256(source)

    output = tmp_path / "out"
    manifest = build_curriculum([SourceSpec.jsonl(source)], output, seed=17)
    first_bytes = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest_again = build_curriculum([SourceSpec.jsonl(source)], output, seed=17)
    second_bytes = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }

    assert manifest == manifest_again
    assert first_bytes == second_bytes
    assert source.read_bytes() == source_before
    assert _sha256(source) == source_hash
    assert manifest["sources"][0]["read_only_verified"] is True
    assert manifest["sources"][0]["sha256_before"] == source_hash
    assert manifest["sources"][0]["sha256_after"] == source_hash
    assert manifest["leakage_audit"]["passed"] is True
    assert manifest["split_lineage_counts"] == {"train": 8, "dev": 1, "test": 1}
    assert manifest["read_coverage_launch_gate"] == {
        "passed": False,
        "blocking_reason": (
            "writable-region post-commit whole-field readback is not "
            "guaranteed by the fixed four-output-tick protocol"
        ),
        "target_supervision_lossless": True,
        "artifact_launch_eligible": False,
    }

    episodes = _read_episodes(output)
    lineages: dict[str, set[str]] = {}
    for episode in episodes:
        lineages.setdefault(episode["source_lineage"], set()).add(episode["split"])
    assert len(lineages) == 10
    assert all(len(splits) == 1 for splits in lineages.values())
    assert {episode["family"] for episode in episodes} == set(families)
    assert all(episode["logical_regions"] == list(CANONICAL_REGIONS) for episode in episodes)


def test_episode_hashes_diffs_masks_and_situation_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            _family_row(FAMILY_GRAMMAR, "grammar-lineage", "one"),
            _family_row(FAMILY_TASK, "task-lineage", "one"),
        ],
    )
    output = tmp_path / "out"
    build_curriculum([SourceSpec.jsonl(source)], output, seed=3)
    episodes = _read_episodes(output)

    grammar = next(episode for episode in episodes if episode["family"] == FAMILY_GRAMMAR)
    assert len(grammar["ticks"]) == 2 * SEGMENT_COUNT
    assert grammar["initial_field_sha256"] == field_sha256(grammar["initial_field"])
    for tick in grammar["ticks"]:
        base = tick["field_before"][tick["target_region"]]["text"]
        segment = tick["segment_target_text"]
        committed = tick["complete_proposed_region_text"]
        delta = tick["typed_delta"]
        assert apply_segment_delta(tick["field_before"], delta) == tick[
            "field_after_teacher_commit"
        ]
        assert len(segment) <= 64
        assert delta["replacement"] == segment
        assert delta["start"] == (0 if tick["segment_index"] == 0 else len(base))
        assert delta["end"] == len(base)
        assert tick["base_region_sha256"] == sha256_text(base)
        assert tick["committed_region_sha256"] == sha256_text(committed)
        assert tick["input_field_sha256"] == field_sha256(tick["field_before"])
        assert tick["teacher_forced"]["committed_field_sha256"] == field_sha256(
            tick["field_after_teacher_commit"]
        )
        assert tick["free_running_evaluation"]["teacher_forcing_allowed"] is False
        projection = tick["active_view"]["proposal_projection"]
        expected_offset = proposal_tail_offset(tick["target_region"], base)
        assert projection["payload_capacity"] == proposal_payload_capacity(
            tick["target_region"]
        )
        assert projection["proposal_view_offset"] == expected_offset
        assert projection["shows_current_tail"] is True
    assert [tick["segment_index"] for tick in grammar["ticks"]] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]

    task = next(episode for episode in episodes if episode["family"] == FAMILY_TASK)
    assert task["initial_field"]["task_state"]["text"] == "active task task-lineage one"
    situation_cell = task["initial_field"]["situation_awareness"]
    assert situation_cell["text"] == "working on task-lineage one"
    assert any(span["type"] == "source_projection" for span in situation_cell["spans"])
    assert task["ticks"][0]["target_region"] == "scratch"
    assert {
        tick["target_region"] for tick in task["ticks"]
    } <= {"scratch", "response_draft"}
    for tick in task["ticks"]:
        assert (
            tick["field_after_teacher_commit"]["task_state"]
            == task["initial_field"]["task_state"]
        )
        assert (
            tick["field_after_teacher_commit"]["situation_awareness"]
            == task["initial_field"]["situation_awareness"]
        )
        assert (
            tick["active_view"]["proposal_projection"]["dynamic_target_region"]
            == tick["target_region"]
        )
        assert tick["active_view"]["context_projection"]["fixed_logical_sub_slices"] is False
    profile = task["field_profile"]
    assert profile["learned_physical_role_count"] == 3
    assert {
        role["physical_type_id"] for role in profile["physical_roles"]
    } == {0, 1, 2}
    assert profile["total_slots"] == N_SLOTS
    assert profile["writable_proposal_regions"] == ["response_draft", "scratch"]


def test_built_ticks_encode_one_persistent_read_cursor_across_region_switch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    row = _family_row(FAMILY_SCRATCH, "paging-lineage", "one")
    row.update(
        {
            "conversation_history": "history " * 60,
            "structured_knowledge": "knowledge " * 55,
            "situation_awareness": "situation " * 50,
            "tool_results": "tool " * 70,
            "advisor_input": "advisor " * 55,
            "task_state": "task " * 70,
            "diary": "diary " * 60,
            "user_input": "question " * 45,
            "initial_draft": "draft " * 20,
            "scratch_plan": "s" * 130,
            "response_after": "r" * 201,
        }
    )
    _write_jsonl(source, [row])
    output = tmp_path / "out"
    build_curriculum(
        [SourceSpec.jsonl(source)],
        output,
        seed=19,
        include_diary=True,
    )
    episode = _read_episodes(output)[0]
    ticks = episode["ticks"]

    assert len(ticks) == 2 * SEGMENT_COUNT
    assert episode["field_profile"]["read_paging"] == {
        "protocol": READ_PAGING_PROTOCOL,
        "cursor_scope": "episode",
        "proposal_switch_resets_cursor": False,
        "reset_condition": "complete_sealed_context_and_user_cycle",
        "proposal_window": "tag_aware_current_region_tail",
    }
    assert [tick["read_page"]["page_index"] for tick in ticks] == list(
        range(2 * SEGMENT_COUNT)
    )
    assert ticks[3]["target_region"] == "scratch"
    assert ticks[4]["target_region"] == "response_draft"
    assert (
        ticks[4]["read_page"]["cursor_before"]
        == ticks[3]["read_page"]["cursor_after"]
    )
    assert all(
        not tick["read_page"]["read_cycle_complete"]
        for tick in ticks
    )
    assert len(
        {tick["read_page"]["read_view_hash"] for tick in ticks}
    ) == len(ticks)
    for tick in ticks:
        page = tick["read_page"]
        context = tick["active_view"]["context_projection"]
        assert page["protocol"] == READ_PAGING_PROTOCOL
        assert len(page["read_view_hash"]) == 64
        assert len(page["coverage"]["refs_sha256"]) == 64
        assert page["coverage"]["duplicate_character_count"] == 0
        assert context["read_page_index"] == page["page_index"]
        assert context["read_view_hash"] == page["read_view_hash"]
        assert context["cursor_before"] == page["cursor_before"]
        assert context["cursor_after"] == page["cursor_after"]


def test_builder_field_contract_matches_runtime_constants() -> None:
    assert CANONICAL_REGIONS == tuple(region.value for region in CANONICAL_REGION_ORDER)
    assert WRITABLE_REGIONS == frozenset(
        region.value for region in CORE_WRITABLE_REGIONS
    )
    assert WRITABLE_REGIONS == frozenset({"scratch", "response_draft"})
    assert PHYSICAL_384_PROFILE == (
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


@pytest.mark.parametrize("region", ["task_state", "situation_awareness", "diary"])
def test_explicit_sealed_targets_fail_closed(tmp_path: Path, region: str) -> None:
    source = tmp_path / f"sealed-{region}.jsonl"
    _write_jsonl(
        source,
        [
            {
                "family": FAMILY_NOOP,
                "lineage_id": "sealed-attempt",
                "initial_draft": "safe draft",
                "targets": [{"region": region, "text": "forbidden overwrite"}],
            }
        ],
    )

    with pytest.raises(SealedTargetError):
        build_curriculum([SourceSpec.jsonl(source)], tmp_path / f"out-{region}")


def test_unsupported_rejection_and_default_diary_exclusion(tmp_path: Path) -> None:
    source = tmp_path / "private_and_unsupported.jsonl"
    private_text = "private continuity note"
    _write_jsonl(
        source,
        [
            {
                "family": FAMILY_GRAMMAR,
                "lineage_id": "unsupported",
                "surface": "plain draft",
                "target": "hot fire \U0001f525",
            },
            {
                "family": FAMILY_DIARY,
                "lineage_id": "diary",
                "diary_text": private_text,
                "target": "continuity response",
            },
            {
                "family": FAMILY_NOOP,
                "lineage_id": "safe",
                "initial_draft": "keep this",
            },
        ],
    )
    source_before = source.read_bytes()
    output = tmp_path / "out"
    manifest = build_curriculum([SourceSpec.jsonl(source)], output)

    assert source.read_bytes() == source_before
    assert manifest["episodes"] == 1
    assert manifest["family_counts"][FAMILY_DIARY] == 0
    assert manifest["skipped"]["unsupported_source_text"] == 1
    assert manifest["skipped"]["diary_family_excluded"] == 1
    written = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert private_text not in written
    assert "\U0001f525" not in written

    diary_output = tmp_path / "diary_out"
    diary_manifest = build_curriculum(
        [SourceSpec.jsonl(source)],
        diary_output,
        include_diary=True,
    )
    diary_episode = next(
        episode for episode in _read_episodes(diary_output) if episode["family"] == FAMILY_DIARY
    )
    assert diary_manifest["diary"]["cloud_export_allowed"] is False
    assert diary_episode["local_only"] is True
    assert diary_episode["privacy"]["diary_included"] is True
    assert {
        tick["target_region"] for tick in diary_episode["ticks"]
    } == {"response_draft"}
    assert all(
        tick["field_after_teacher_commit"]["diary"]
        == diary_episode["initial_field"]["diary"]
        for tick in diary_episode["ticks"]
    )


def test_targets_use_four_segments_and_over_budget_is_explicitly_excluded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized.jsonl"
    _write_jsonl(
        source,
        [
            {
                "family": FAMILY_GRAMMAR,
                "lineage_id": "sixty-five",
                "surface": "short draft",
                "target": "x" * 65,
            },
            {
                "family": FAMILY_GRAMMAR,
                "lineage_id": "two-hundred-one",
                "surface": "short draft",
                "target": "y" * 201,
            },
            {
                "family": FAMILY_GRAMMAR,
                "lineage_id": "over-budget",
                "surface": "short draft",
                "target": "z" * (REGION_TARGET_BUDGET + 1),
            },
            {
                "family": FAMILY_GRAMMAR,
                "lineage_id": "bounded",
                "surface": "teh bounded answer",
                "target": "the bounded answer",
            },
        ],
    )

    output = tmp_path / "out"
    manifest = build_curriculum([SourceSpec.jsonl(source)], output)
    episodes = _read_episodes(output)

    assert manifest["episodes"] == 3
    assert manifest["skipped"]["proposal_target_exceeds_256"] == 1
    assert not any(
        episode["source_lineage"].endswith(":over-budget")
        for episode in episodes
    )
    for episode in episodes:
        assert len(episode["ticks"]) == 2 * SEGMENT_COUNT
        assert all(
            len(tick["segment_target_text"]) <= 64
            for tick in episode["ticks"]
        )
        for transition_index in range(2):
            transition = [
                tick
                for tick in episode["ticks"]
                if tick["transition_index"] == transition_index
            ]
            assert [tick["segment_index"] for tick in transition] == [0, 1, 2, 3]
    sixty_five = next(
        episode
        for episode in episodes
        if episode["source_lineage"].endswith(":sixty-five")
    )
    two_hundred_one = next(
        episode
        for episode in episodes
        if episode["source_lineage"].endswith(":two-hundred-one")
    )
    assert [
        len(tick["segment_target_text"])
        for tick in sixty_five["ticks"][-SEGMENT_COUNT:]
    ] == [64, 1, 0, 0]
    assert [
        len(tick["segment_target_text"])
        for tick in two_hundred_one["ticks"][-SEGMENT_COUNT:]
    ] == [64, 64, 64, 9]
    assert sixty_five["ticks"][-1]["complete_proposed_region_text"] == "x" * 65
    assert (
        two_hundred_one["ticks"][-1]["complete_proposed_region_text"]
        == "y" * 201
    )


def test_lossless_escape_policy_has_audited_inverse() -> None:
    original = "literal [ bracket and fire \U0001f525"
    encoder = ExactTextEncoder("escape")
    encoded = encoder.encode(original, "test.field")

    assert encoded != original
    assert_supported_text(encoded)
    assert decode_lossless_escapes(encoded) == original
    assert encoder.audits[0]["policy"] == "lossless_codepoint_escape"
    assert {item["codepoint"] for item in encoder.audits[0]["mappings"]} == {
        "U+005B",
        "U+1F525",
    }


def test_sqlite_adapter_uses_uri_mode_ro_and_does_not_mutate_source(tmp_path: Path) -> None:
    database = tmp_path / "source.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "create table curriculum ("
            "id integer primary key, family text, lineage_id text, "
            "surface text, target text)"
        )
        connection.execute(
            "insert into curriculum (family, lineage_id, surface, target) values (?, ?, ?, ?)",
            (FAMILY_GRAMMAR, "sqlite-lineage", "teh sqlite row", "the sqlite row"),
        )
        connection.commit()
    finally:
        connection.close()

    before_hash = _sha256(database)
    before_mtime = database.stat().st_mtime_ns
    uri = sqlite_ro_uri(database)
    assert "mode=ro" in uri

    output = tmp_path / "out"
    manifest = build_curriculum(
        [SourceSpec.sqlite(database, table="curriculum")],
        output,
    )

    assert _sha256(database) == before_hash
    assert database.stat().st_mtime_ns == before_mtime
    assert manifest["sources"][0]["kind"] == "sqlite"
    assert manifest["sources"][0]["read_only_verified"] is True
    assert manifest["episodes"] == 1
    assert _read_episodes(output)[0]["provenance"]["record_pointer"].startswith("sqlite:")


def test_cross_split_exact_and_near_duplicate_audit() -> None:
    base = {
        "episode_id": "a",
        "source_lineage": "lineage-a",
        "split": "train",
        "initial_field": {
            region: {
                "text": "Same Content!" if region == "user_input" else "",
            }
            for region in CANONICAL_REGIONS
        },
        "ticks": [
            {
                "target_region": "response_draft",
                "complete_proposed_region_text": "Answer One.",
            }
        ],
    }
    near = copy.deepcopy(base)
    near["episode_id"] = "b"
    near["source_lineage"] = "lineage-b"
    near["split"] = "test"
    near["initial_field"]["user_input"]["text"] = "same content"
    near["ticks"][0]["complete_proposed_region_text"] = "answer one"

    report = audit_cross_split_leakage([base, near])
    assert report["passed"] is False
    assert not report["exact_cross_split"]
    assert report["near_cross_split"]


def test_duplicate_connected_lineages_are_transitively_co_located(
    tmp_path: Path,
) -> None:
    source = tmp_path / "duplicate-components.jsonl"
    rows = [
        {
            "family": FAMILY_NOOP,
            "lineage_id": "near-a",
            "initial_draft": "Alpha One!",
        },
        {
            "family": FAMILY_NOOP,
            "lineage_id": "near-bridge",
            "initial_draft": "alpha one",
        },
        {
            "family": FAMILY_NOOP,
            "lineage_id": "near-bridge",
            "initial_draft": "Beta Two!",
        },
        {
            "family": FAMILY_NOOP,
            "lineage_id": "near-c",
            "initial_draft": "beta two",
        },
        {
            "family": FAMILY_NOOP,
            "lineage_id": "exact-a",
            "initial_draft": "Exact duplicate content",
        },
        {
            "family": FAMILY_NOOP,
            "lineage_id": "exact-b",
            "initial_draft": "Exact duplicate content",
        },
    ]
    rows.extend(
        {
            "family": FAMILY_NOOP,
            "lineage_id": f"unique-{index}",
            "initial_draft": f"unique content number {index}",
        }
        for index in range(6)
    )
    _write_jsonl(source, rows)

    output = tmp_path / "out"
    manifest = build_curriculum([SourceSpec.jsonl(source)], output, seed=1)
    episodes = _read_episodes(output)

    def lineage_episodes(name: str) -> list[dict]:
        return [
            episode
            for episode in episodes
            if episode["source_lineage"].endswith(f":{name}")
        ]

    transitive = [
        episode
        for name in ("near-a", "near-bridge", "near-c")
        for episode in lineage_episodes(name)
    ]
    assert len(transitive) == 4
    assert len({episode["split"] for episode in transitive}) == 1
    assert len({episode["split_cluster_id"] for episode in transitive}) == 1
    assert {episode["split_cluster_lineage_count"] for episode in transitive} == {3}

    exact = [
        episode
        for name in ("exact-a", "exact-b")
        for episode in lineage_episodes(name)
    ]
    assert len(exact) == 2
    assert len({episode["split"] for episode in exact}) == 1
    assert len({episode["split_cluster_id"] for episode in exact}) == 1
    assert {episode["split_cluster_lineage_count"] for episode in exact} == {2}
    assert manifest["leakage_audit"]["passed"] is True
    assert manifest["split_cluster_count"] == 8
