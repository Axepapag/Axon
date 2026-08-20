from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from runtime.field import CANONICAL_REGION_ORDER
from substrate import default_alphabet, get_letter_bank
import training.build_conversational_curriculum as curriculum_module
from training.build_conversational_curriculum import (
    CANONICAL_REGIONS,
    EXAMPLE_SCHEMA,
    MANIFEST_SCHEMA,
    SOURCE_SCHEMA,
    CurriculumContractError,
    build_conversational_curriculum,
    canonical_json_bytes,
    load_source_records,
    source_record_sha256,
)


ROOT = Path(__file__).resolve().parent.parent
REGION_NAMES = (
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
)


def _source(
    source_id: str,
    conversation: list[dict[str, str]],
) -> dict:
    record = {
        "schema": SOURCE_SCHEMA,
        "source_id": source_id,
        "source_sha256": "",
        "license": {
            "spdx_id": "CC-BY-4.0",
            "license_url": (
                "https://creativecommons.org/licenses/by/4.0/"
            ),
            "redistribution_allowed": True,
            "derivatives_allowed": True,
            "commercial_use_allowed": True,
        },
        "provenance": {
            "publisher": "Public Example Foundation",
            "source_name": "Public conversation fixture",
            "source_url": f"https://example.org/conversations/{source_id}",
            "retrieved_utc": "2026-07-28T00:00:00Z",
        },
        "privacy": {
            "classification": "public",
            "local_only": False,
            "cloud_export_allowed": True,
            "contains_personal_data": False,
        },
        "conversation": conversation,
    }
    record["source_sha256"] = source_record_sha256(record)
    return record


def _rehash(record: dict) -> dict:
    record["source_sha256"] = source_record_sha256(record)
    return record


def _write_sources(path: Path, records: list[dict]) -> bytes:
    payload = b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    )
    path.write_bytes(payload)
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_build_is_deterministic_source_split_and_ten_region_contract(
    tmp_path: Path,
) -> None:
    assert tuple(region.value for region in CANONICAL_REGION_ORDER) == REGION_NAMES
    assert CANONICAL_REGIONS == REGION_NAMES
    assert len(REGION_NAMES) == 10
    source_a = _source(
        "public-a",
        [
            {"role": "user", "text": "Give the first answer."},
            {"role": "assistant", "text": "The first answer is ready."},
            {"role": "user", "text": "Should the draft change now?"},
            {
                "role": "assistant",
                "kind": "structural_noop",
                "text": "No draft update is required.",
            },
        ],
    )
    source_b = _source(
        "public-b",
        [
            {
                "role": "user",
                "text": "Read line one.\nThen read line two!",
            },
            {
                "role": "assistant",
                "text": "Both public lines were read.",
            },
        ],
    )
    input_path = tmp_path / "sources.jsonl"
    input_bytes = _write_sources(input_path, [source_b, source_a])
    before_mtime = input_path.stat().st_mtime_ns
    output_a = tmp_path / "examples-a.jsonl"
    output_b = tmp_path / "examples-b.jsonl"
    manifest_a_path = tmp_path / "manifest-a.json"
    manifest_b_path = tmp_path / "manifest-b.json"

    manifest_a = build_conversational_curriculum(
        input_path,
        output_a,
        manifest_a_path,
    )
    manifest_b = build_conversational_curriculum(
        input_path,
        output_b,
        manifest_b_path,
    )

    assert output_a.read_bytes() == output_b.read_bytes()
    assert manifest_a == manifest_b
    assert manifest_a_path.read_bytes() == manifest_b_path.read_bytes()
    assert input_path.read_bytes() == input_bytes
    assert input_path.stat().st_mtime_ns == before_mtime

    examples = _read_jsonl(output_a)
    assert len(examples) == 3
    assert all(example["schema"] == EXAMPLE_SCHEMA for example in examples)
    by_source: dict[str, list[dict]] = {}
    for example in examples:
        by_source.setdefault(example["source"]["source_id"], []).append(example)
        assert list(example["active_field"]) == sorted(REGION_NAMES)
        assert set(example["active_field"]) == set(REGION_NAMES)
        assert example["active_field"]["response_draft"] == ""
        assert example["gold_exclusion_audit"]["passed"] is True
        if example["target_delta"] is not None:
            target = example["target_delta"]["text"]
            assert 1 <= len(target) <= 64
            assert example["target_delta"] == {
                "char_count": len(target),
                "complete_replacement": True,
                "op": "replace",
                "region": "response_draft",
                "text": target,
            }
            assert all(
                target not in visible
                for visible in example["active_field"].values()
            )

    assert {item["split"] for item in by_source["public-a"]} == {
        by_source["public-a"][0]["split"]
    }
    second_a = max(
        by_source["public-a"],
        key=lambda item: item["source"]["assistant_turn_index"],
    )
    assert (
        second_a["active_field"]["conversation_history"]
        == "User: Give the first answer.\n"
        "Assistant: The first answer is ready."
    )
    assert second_a["active_field"]["user_input"] == (
        "Should the draft change now?"
    )
    assert second_a["example_kind"] == "structural_noop"
    assert second_a["target_delta"] is None
    assert second_a["structural_noop"] == {
        "op": "no_op",
        "reason": "No draft update is required.",
        "reason_char_count": 28,
    }
    assert second_a["structural_noop"]["reason"]

    assert manifest_a["schema"] == MANIFEST_SCHEMA
    assert manifest_a["realized_counts"] == {
        "sources": 2,
        "conversations": 2,
        "examples": 3,
        "response_draft_replacements": 2,
        "structural_noops": 1,
        "sources_by_split": manifest_a["realized_counts"][
            "sources_by_split"
        ],
        "examples_by_split": manifest_a["realized_counts"][
            "examples_by_split"
        ],
    }
    assert sum(
        manifest_a["realized_counts"]["sources_by_split"].values()
    ) == 2
    assert sum(
        manifest_a["realized_counts"]["examples_by_split"].values()
    ) == 3
    assert manifest_a["artifacts"]["input_jsonl"] == {
        "bytes": len(input_bytes),
        "sha256": hashlib.sha256(input_bytes).hexdigest().upper(),
    }
    assert manifest_a["artifacts"]["examples_jsonl"]["sha256"] == (
        hashlib.sha256(output_a.read_bytes()).hexdigest().upper()
    )
    payload = dict(manifest_a)
    payload_hash = payload.pop("manifest_payload_sha256")
    assert payload_hash == hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest().upper()
    assert {
        item["source_sha256"] for item in manifest_a["sources"]
    } == {source_a["source_sha256"], source_b["source_sha256"]}
    alphabet = list(default_alphabet())
    bank = get_letter_bank()
    assert manifest_a["example_contract"]["substrate"] == {
        "character_count": 95,
        "ordered_alphabet_sha256": hashlib.sha256(
            canonical_json_bytes(alphabet)
        ).hexdigest().upper(),
        "decoder_class_count": 96,
        "ordered_letter_bank_sha256": hashlib.sha256(
            canonical_json_bytes(list(bank.chars))
        ).hexdigest().upper(),
        "empty_index": 95,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record["privacy"].__setitem__(
                "local_only",
                True,
            ),
            "local_only",
        ),
        (
            lambda record: record["privacy"].__setitem__(
                "cloud_export_allowed",
                False,
            ),
            "cloud_export_allowed=false",
        ),
        (
            lambda record: record["privacy"].__setitem__(
                "classification",
                "private",
            ),
            "classification",
        ),
        (
            lambda record: record["privacy"].__setitem__(
                "contains_personal_data",
                True,
            ),
            "contains_personal_data=false",
        ),
        (
            lambda record: record["license"].__setitem__(
                "redistribution_allowed",
                False,
            ),
            "redistribution",
        ),
        (
            lambda record: record["license"].__setitem__(
                "derivatives_allowed",
                False,
            ),
            "derivative",
        ),
        (
            lambda record: record["license"].__setitem__(
                "commercial_use_allowed",
                False,
            ),
            "commercial use",
        ),
        (
            lambda record: record["license"].__setitem__(
                "spdx_id",
                "NOASSERTION",
            ),
            "exportable-license allowlist",
        ),
        (
            lambda record: record["provenance"].__setitem__(
                "source_url",
                "https://intranet/data",
            ),
            "single-label private host",
        ),
        (
            lambda record: record["provenance"].__setitem__(
                "source_url",
                "https://user:secret@example.org/data",
            ),
            "embedded credentials",
        ),
        (
            lambda record: record["provenance"].__setitem__(
                "source_url",
                "https://127.1/data",
            ),
            "noncanonical numeric host",
        ),
        (
            lambda record: record["provenance"].__setitem__(
                "source_url",
                "https://0x7f.0.0.1/data",
            ),
            "noncanonical numeric host",
        ),
    ],
)
def test_rejects_non_public_or_non_exportable_sources(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    record = _source(
        "restricted",
        [
            {"role": "user", "text": "A public question."},
            {"role": "assistant", "text": "A public answer."},
        ],
    )
    mutation(record)
    _rehash(record)
    input_path = tmp_path / "restricted.jsonl"
    _write_sources(input_path, [record])

    with pytest.raises(CurriculumContractError, match=message):
        build_conversational_curriculum(
            input_path,
            tmp_path / "must-not-exist.jsonl",
            tmp_path / "must-not-exist.manifest.json",
        )

    assert not (tmp_path / "must-not-exist.jsonl").exists()
    assert not (tmp_path / "must-not-exist.manifest.json").exists()


@pytest.mark.parametrize(
    "provenance_mutation",
    [
        lambda provenance: provenance.__setitem__(
            "source_name",
            r"D:\00\private-memory.jsonl",
        ),
        lambda provenance: provenance.__setitem__(
            "local_path",
            r"C:\Users\someone\private.jsonl",
        ),
        lambda provenance: provenance.__setitem__(
            "source_url",
            "file:///D:/00/private-memory.jsonl",
        ),
        lambda provenance: provenance.__setitem__(
            "source_name",
            r"..\..\00\private-memory.jsonl",
        ),
    ],
)
def test_rejects_d00_and_private_path_provenance(
    tmp_path: Path,
    provenance_mutation,
) -> None:
    record = _source(
        "private-path",
        [
            {"role": "user", "text": "Question."},
            {"role": "assistant", "text": "Answer."},
        ],
    )
    provenance_mutation(record["provenance"])
    _rehash(record)
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])

    with pytest.raises(
        CurriculumContractError,
        match="private or local path provenance",
    ):
        build_conversational_curriculum(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    "private_path",
    [
        r"D:\00\private-source.jsonl",
        r"D:\Axon\..\00\private-source.jsonl",
    ],
)
def test_d00_input_is_rejected_before_any_read(private_path: str) -> None:
    with pytest.raises(CurriculumContractError, match=r"D:\\00"):
        load_source_records(private_path)


def test_exact_substrate_rejects_without_substitution_or_partial_output(
    tmp_path: Path,
) -> None:
    record = _source(
        "unsupported",
        [
            {"role": "user", "text": "Keep this exact."},
            {"role": "assistant", "text": "Never replace this emoji."},
        ],
    )
    record["conversation"][1]["text"] = "Never replace this emoji: \U0001f642"
    _rehash(record)
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])
    output = tmp_path / "out.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(
        CurriculumContractError,
        match="exact 95-character substrate",
    ):
        build_conversational_curriculum(source, output, manifest)

    assert not output.exists()
    assert not manifest.exists()


def test_complete_target_over_64_is_rejected_not_chunked(
    tmp_path: Path,
) -> None:
    record = _source(
        "too-long",
        [
            {"role": "user", "text": "Give one complete replacement."},
            {"role": "assistant", "text": "a" * 65},
        ],
    )
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])

    with pytest.raises(CurriculumContractError, match="complete target limit"):
        build_conversational_curriculum(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
        )


@pytest.mark.parametrize("kind", ["response", "structural_noop"])
def test_empty_response_or_noop_reason_is_never_an_answer_target(
    tmp_path: Path,
    kind: str,
) -> None:
    record = _source(
        f"empty-{kind}",
        [
            {"role": "user", "text": "Do something."},
            {"role": "assistant", "kind": kind, "text": ""},
        ],
    )
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])

    with pytest.raises(CurriculumContractError, match="non-empty"):
        build_conversational_curriculum(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
        )


def test_visible_gold_collision_fails_closed(tmp_path: Path) -> None:
    record = _source(
        "repeat",
        [
            {"role": "user", "text": "First request."},
            {"role": "assistant", "text": "Repeat."},
            {"role": "user", "text": "Second request."},
            {"role": "assistant", "text": "Repeat."},
        ],
    )
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])

    with pytest.raises(
        CurriculumContractError,
        match="gold target appears in visible field",
    ):
        build_conversational_curriculum(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
        )


def test_source_hash_is_verified(tmp_path: Path) -> None:
    record = _source(
        "hash-check",
        [
            {"role": "user", "text": "Question."},
            {"role": "assistant", "text": "Answer."},
        ],
    )
    record["source_sha256"] = "0" * 64
    source = tmp_path / "source.jsonl"
    _write_sources(source, [record])

    with pytest.raises(CurriculumContractError, match="source_sha256 mismatch"):
        build_conversational_curriculum(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
        )


def test_caller_controlled_legacy_temp_name_cannot_alias_input(
    tmp_path: Path,
) -> None:
    record = _source(
        "temp-alias",
        [
            {"role": "user", "text": "Preserve the source."},
            {"role": "assistant", "text": "The source remains intact."},
        ],
    )
    output = tmp_path / "examples.jsonl"
    source = tmp_path / ".examples.jsonl.axon-tmp"
    manifest = tmp_path / "manifest.json"
    original = _write_sources(source, [record])

    build_conversational_curriculum(source, output, manifest)

    assert source.exists()
    assert source.read_bytes() == original
    assert output.exists()
    assert manifest.exists()


@pytest.mark.parametrize("existing_pair", [False, True])
def test_output_and_manifest_roll_back_as_a_pair_on_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_pair: bool,
) -> None:
    record = _source(
        "pair-rollback",
        [
            {"role": "user", "text": "Build both artifacts."},
            {"role": "assistant", "text": "Both artifacts are required."},
        ],
    )
    source = tmp_path / "source.jsonl"
    original_source = _write_sources(source, [record])
    output = tmp_path / "examples.jsonl"
    manifest = tmp_path / "manifest.json"
    if existing_pair:
        output.write_bytes(b"old examples\n")
        manifest.write_bytes(b"old manifest\n")

    real_replace = curriculum_module.os.replace
    replace_count = 0
    fail_at = 4 if existing_pair else 2

    def fail_second_install(source_path, destination_path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == fail_at:
            raise OSError("synthetic second-artifact install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(curriculum_module.os, "replace", fail_second_install)

    with pytest.raises(
        OSError,
        match="synthetic second-artifact install failure",
    ):
        build_conversational_curriculum(source, output, manifest)

    assert source.read_bytes() == original_source
    if existing_pair:
        assert output.read_bytes() == b"old examples\n"
        assert manifest.read_bytes() == b"old manifest\n"
    else:
        assert not output.exists()
        assert not manifest.exists()
    assert not list(tmp_path.glob("*.axon-tmp-*"))
    assert not list(tmp_path.glob("*.axon-backup-*"))


def test_import_has_no_artifact_writes(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import training.build_conversational_curriculum",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
