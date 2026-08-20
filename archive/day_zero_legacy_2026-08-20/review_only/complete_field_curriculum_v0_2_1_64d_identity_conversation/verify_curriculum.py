#!/usr/bin/env python3
"""Independent semantic and deterministic-replay verifier for Axon v0.2.1.

The verifier deliberately does not import the builder.  It recomputes record,
snapshot, envelope, coverage, authority, chunk, and distribution invariants
from emitted bytes.  It may launch two isolated builder runs when requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERIFIER_VERSION = "axon-complete-field-verifier-v0.2"
REGIONS = (
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
IMMUTABLE_CONTENT_REGIONS = frozenset(
    {"conversation_history", "user_input", "tool_results", "advisor_input"}
)
FOCUS_PROFILE = "core-r0-64d-identity-conversation"
FOCUS_WRITABLE_REGIONS = {"scratch", "response_draft", "diary"}
PHASE_TO_SEQ = {"proposal": 0, "refinement": 1, "consolidation": 2}
REQUIRED_VARIANTS = {
    "correct",
    "missing_page",
    "duplicated_page",
    "reordered_page",
    "hash_corrupted",
}
REQUIRED_BINS = {"first", "early", "middle", "late", "last"}
REQUIRED_STRATA = {1, 2, 4, 8}


class VerificationError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_canonical(value: Any) -> str:
    return sha256_text(canonical_json(value))


def digest_text(value: str) -> dict[str, str]:
    return {"algorithm": "sha256", "digest": sha256_text(value)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            require(isinstance(value, dict), f"{path.name}:{line_number}: row is not an object")
            records.append(value)
    return records


def region_text(snapshot: dict[str, Any], name: str) -> str:
    matches = [region for region in snapshot["regions"] if region["name"] == name]
    require(len(matches) == 1, f"snapshot does not contain exactly one {name} region")
    return "".join(span["text"] for span in matches[0]["spans"])


def is_allowed(
    profiles: dict[str, Any],
    profile_id: str,
    core_id: str,
    action_class: str,
    region: str,
    operation: str,
) -> bool:
    if action_class == "content" and region in IMMUTABLE_CONTENT_REGIONS:
        return False
    profile = profiles["profiles"].get(profile_id, {})
    core = profile.get("cores", {}).get(core_id, {})
    return operation in core.get(action_class, {}).get(region, [])


def validate_profiles(profiles: dict[str, Any]) -> None:
    require(profiles["schema"] == "axon-core-authority-profiles-v0.2", "wrong authority profile schema")
    require(profiles["default_decision"] == "deny", "authority must default deny")
    require(
        set(profiles["content_immutable_to_cores"]) == IMMUTABLE_CONTENT_REGIONS,
        "immutable core content regions changed",
    )
    require(
        set(profiles["profiles"]) == {
            "core-v1-scratch-response-only",
            "core-v2-governed-full-field",
            FOCUS_PROFILE,
        },
        "required policy profiles missing",
    )
    for profile_id, profile in profiles["profiles"].items():
        for core_id, core in profile["cores"].items():
            for region in IMMUTABLE_CONTENT_REGIONS:
                require(
                    not core["content"].get(region),
                    f"{profile_id}/{core_id} grants immutable content region {region}",
                )
            for region, operations in core["attention"].items():
                require(region in REGIONS, f"unknown attention region {region}")
                require("set_intervals" not in operations, "R0 enables future interval masks")
    v1 = profiles["profiles"]["core-v1-scratch-response-only"]
    for core_id, core in v1["cores"].items():
        require(set(core["content"]) <= {"scratch", "response_draft"}, f"v1 core {core_id} writes outside compatibility regions")
        require(not core["attention"], f"v1 core {core_id} unexpectedly owns the canonical mask")
    focus = profiles["profiles"][FOCUS_PROFILE]
    require(focus["mask_scope"] == "core_private_attention_view", "focus profile mask is not core-private")
    for core_id, core in focus["cores"].items():
        require(set(core["content"]) == FOCUS_WRITABLE_REGIONS, f"focus core {core_id} has the wrong content regions")
        require(set(core["attention"]) == set(REGIONS), f"focus core {core_id} cannot attend every region")
        require(
            all("set_boundary" in core["attention"][region] for region in REGIONS),
            f"focus core {core_id} lacks a regional boundary switch",
        )


def validate_alphabet(alphabet: dict[str, Any]) -> set[str]:
    require(alphabet["alphabet_id"] == "axon-char-alphabet-v8", "wrong alphabet id")
    require(alphabet["algorithm"] == "sha256", "wrong alphabet digest algorithm")
    require(len(alphabet["ordered_characters"]) == alphabet["length"] == 95, "alphabet length mismatch")
    require(
        sha256_text(canonical_json(alphabet["ordered_characters"])) == alphabet["digest"],
        "alphabet canonical digest mismatch",
    )
    require(
        sha256_text("".join(alphabet["ordered_characters"])) == alphabet["raw_joined_digest"],
        "alphabet raw joined digest mismatch",
    )
    return set(alphabet["ordered_characters"])


def independent_namespace_oracle(namespace: str, value: dict[str, Any]) -> str | None:
    prefix = "BASE_FIELD" if namespace == "base_field" else "SIBLING_DELTA_SET"
    pages = value["pages"]
    if not value["required"]:
        if pages or value["expected_characters"]:
            return f"{prefix}_UNEXPECTED_STREAM"
        return None
    indices = [page["page_index"] for page in pages]
    expected = value["expected_page_indices"]
    if len(indices) != len(set(indices)):
        return f"{prefix}_DUPLICATE_PAGE"
    if set(indices) != set(expected):
        if set(indices).issubset(set(expected)):
            return f"{prefix}_MISSING_PAGE"
        return f"{prefix}_PAGE_INDEX_INVALID"
    if indices != expected:
        return f"{prefix}_REORDERED_PAGE"
    cursor = 0
    text = []
    for page in pages:
        if page["namespace"] != namespace or page["stream_id"] != value["stream_id"]:
            return f"{prefix}_STREAM_ID_MISMATCH"
        if page["stream_start"] != cursor or page["stream_end"] < cursor:
            return f"{prefix}_COVERAGE_GAP"
        if page["digest"] != digest_text(page["exact_text"]):
            return f"{prefix}_HASH_MISMATCH"
        cursor = page["stream_end"]
        text.append(page["exact_text"])
    assembled = "".join(text)
    if cursor != value["expected_characters"]:
        return f"{prefix}_COVERAGE_GAP"
    if value["source_digest"] != digest_text(assembled):
        return f"{prefix}_SOURCE_DIGEST_MISMATCH"
    return None


def independent_coverage_oracle(manifest: dict[str, Any]) -> str | None:
    for namespace in ("base_field", "sibling_delta_set"):
        failure = independent_namespace_oracle(namespace, manifest[namespace])
        if failure:
            return failure
    return None


def validate_envelope(envelope: dict[str, Any], snapshot: dict[str, Any]) -> None:
    basis = {key: value for key, value in envelope.items() if key != "envelope_id"}
    require(envelope["envelope_id"] == sha256_canonical(basis), "envelope identity mismatch")
    require(envelope["base_field_id"] == snapshot["field_id"], "envelope base field mismatch")
    require(envelope["base_tick_id"] == snapshot["tick_id"], "envelope base tick mismatch")
    require(envelope["phase_seq"] == PHASE_TO_SEQ[envelope["source_phase"]], "envelope phase mapping mismatch")
    payload = envelope["payload"]
    require(payload["encoding"] == "canonical-json-utf8", "wrong envelope payload encoding")
    require(payload["digest"] == digest_text(payload["exact_text"]), "envelope payload digest mismatch")
    require(payload["delta_id"] == payload["digest"]["digest"], "delta id differs from payload digest")
    delta = json.loads(payload["exact_text"])
    require(canonical_json(delta) == payload["exact_text"], "delta payload is not canonical JSON")
    require(delta["schema"] == "shared-field-delta-v1", "wrong delta payload schema")
    require(delta["base_field_id"] == envelope["base_field_id"], "delta/envelope field disagreement")
    require(delta["base_tick_id"] == envelope["base_tick_id"], "delta/envelope tick disagreement")
    require(delta["author_core_id"] == envelope["core_id"], "delta/envelope core disagreement")
    require(delta["operations"], "FieldDelta payload is empty")
    for operation in delta["operations"]:
        require(operation["op"] in {"insert", "delete", "replace"}, "non-runtime semantic delta operation")
        require(operation["region"] in REGIONS, "delta addresses unknown region")


def validate_sibling_set(record: dict[str, Any]) -> None:
    sibling_set = record["sibling_delta_set"]
    basis = {key: value for key, value in sibling_set.items() if key != "set_id"}
    require(sibling_set["set_id"] == sha256_canonical(basis), "sibling set identity mismatch")
    tick = record["tick"]
    require(sibling_set["tick_seq"] == tick["tick_seq"], "sibling set tick mismatch")
    expected = tick["expected_sibling_core_ids"]
    require(sibling_set["expected_core_ids"] == expected, "record/sibling expected roster mismatch")
    cores = [envelope["core_id"] for envelope in sibling_set["envelopes"]]
    require(len(cores) == len(set(cores)), "duplicate sibling core")
    require(cores == expected, "sibling envelope roster is incomplete or reordered")
    if tick["phase"] == "proposal":
        require(sibling_set["source_phase"] is None and not cores, "proposal phase has sibling deltas")
    else:
        expected_phase = "proposal" if tick["phase"] == "refinement" else "refinement"
        require(sibling_set["source_phase"] == expected_phase, "wrong sibling source phase")
        for roster_index, envelope in enumerate(sibling_set["envelopes"]):
            require(envelope["roster_index"] == roster_index, "sibling roster index mismatch")
            require(envelope["source_phase"] == expected_phase, "sibling envelope phase mismatch")
            validate_envelope(envelope, record["snapshot"])


def reconstruct_active_stream(record: dict[str, Any]) -> str:
    snapshot = record["snapshot"]
    view = record["attention_view"]
    text = []
    for region in REGIONS:
        source = region_text(snapshot, region)
        mask = view["regions"][region]
        require(mask["source_text_digest"] == digest_text(source), f"{region} attention source hash mismatch")
        intervals = mask["active_intervals"]
        previous_end = 0
        for start, end in intervals:
            require(0 <= start < end <= len(source), f"invalid {region} active interval")
            require(start >= previous_end, f"overlapping {region} active intervals")
            previous_end = end
            text.append(source[start:end])
        require(mask["future_intervals"] is None, "future intervals implemented in R0 record")
        require(mask["can_set_intervals"] is False, "future interval switch enabled")
    return "".join(text)


def validate_transport(record: dict[str, Any]) -> None:
    result = record["phase_result"]
    if result["outcome"] != "delta":
        require(result["delta_envelope"] is None, "failed/no-change phase carries a delta envelope")
        require(result["delta_transport"] is None, "failed/no-change phase carries transport chunks")
        return
    envelope = result["delta_envelope"]
    validate_envelope(envelope, record["snapshot"])
    transport = result["delta_transport"]
    chunks = transport["chunks"]
    require(chunks, "delta transport has no chunks")
    expected_count = len(chunks)
    require([chunk["chunk_index"] for chunk in chunks] == list(range(expected_count)), "chunk indices are not ordered and gap-free")
    assembled = []
    for chunk in chunks:
        require(chunk["transaction_id"] == transport["transaction_id"], "chunk transaction mismatch")
        require(chunk["expected_chunk_count"] == expected_count, "chunk count binding mismatch")
        require(chunk["byte_length"] == len(chunk["exact_text"].encode("utf-8")), "chunk byte length mismatch")
        require(chunk["digest"] == digest_text(chunk["exact_text"]), "chunk digest mismatch")
        assembled.append(chunk["exact_text"])
    exact = "".join(assembled)
    require(transport["assembled_digest"] == digest_text(exact), "assembled transport digest mismatch")
    require(exact == canonical_json(envelope), "assembled chunks are not the canonical envelope")


def validate_authority(record: dict[str, Any], profiles: dict[str, Any]) -> None:
    authority = record["authority"]
    require(authority["default_decision"] == "deny", "record authority is not default deny")
    require(authority["core_id"] == record["tick"]["acting_core_id"], "authority core differs from acting core")
    for assertion in authority["policy_assertions"]:
        observed = is_allowed(
            profiles,
            authority["profile_id"],
            authority["core_id"],
            assertion["action_class"],
            assertion["region"],
            assertion["operation"],
        )
        require(observed == assertion["expected_authorized"], f"authority assertion mismatch for {assertion}")
    result = record["phase_result"]
    if result["outcome"] == "delta":
        delta = json.loads(result["delta_envelope"]["payload"]["exact_text"])
        if authority["profile_id"] == FOCUS_PROFILE:
            require(
                {operation["region"] for operation in delta["operations"]}
                == FOCUS_WRITABLE_REGIONS,
                "focus delta does not update exactly scratch, response_draft, and diary",
            )
        for operation in delta["operations"]:
            require(operation["region"] not in IMMUTABLE_CONTENT_REGIONS, "target delta mutates immutable evidence")
            require(
                is_allowed(
                    profiles,
                    authority["profile_id"],
                    authority["core_id"],
                    "content",
                    operation["region"],
                    operation["op"],
                ),
                "target delta bypasses a disabled per-core region switch",
            )
    update = result["attention_update"]
    if update is not None:
        require(
            is_allowed(
                profiles,
                authority["profile_id"],
                authority["core_id"],
                "attention",
                update["region"],
                update["operation"],
            ),
            "attention update bypasses a disabled switch",
        )
        source = region_text(record["snapshot"], update["region"])
        require(update["source_text_digest"] == digest_text(source), "attention update source text changed")
        require(0 <= update["old_offset"] <= len(source), "attention old offset out of range")
        require(0 <= update["new_offset"] <= len(source), "attention new offset out of range")
        require(update["effective"] == "next_tick", "mask update retroactively changes current coverage")
        require(update["scope"] == "core_private_attention_view", "v2 mask is not core-private")


def validate_record(
    record: dict[str, Any], profiles: dict[str, Any], alphabet_chars: set[str]
) -> None:
    required = {
        "record_type", "schema_version", "builder_version", "stamp", "example_id",
        "family", "split", "lineage_id", "counterfactual_group", "counterfactual_variant",
        "tick", "snapshot", "attention_view", "authority", "sibling_delta_set",
        "coverage_manifest", "phase_result", "expected", "evidence_spans",
        "soul_transition", "provenance", "alphabet", "builder_validation",
    }
    require(required <= set(record), f"{record.get('example_id', '<unknown>')} missing required fields")
    require(record["record_type"] == "complete_field_example", "wrong record type")
    require(record["schema_version"] == "0.2.0-review", "wrong record schema version")
    tick = record["tick"]
    require(tick["phase_seq"] == PHASE_TO_SEQ[tick["phase"]], "tick phase mapping mismatch")
    require(tick["acting_core_id"] in tick["online_core_ids"], "acting core is not online")
    require(tick["consolidator_core_id"] in tick["online_core_ids"], "consolidator is not online")
    if tick["phase"] == "consolidation":
        require(tick["acting_core_id"] == tick["consolidator_core_id"], "non-crown core consolidates")

    snapshot = record["snapshot"]
    require([region["name"] for region in snapshot["regions"]] == list(REGIONS), "canonical region order mismatch")
    snapshot_basis = {key: value for key, value in snapshot.items() if key != "field_id"}
    require(snapshot["field_id"] == sha256_canonical(snapshot_basis), "snapshot field identity mismatch")
    require(snapshot["tick_id"] == tick["tick_seq"], "snapshot/tick mismatch")
    require(record["attention_view"]["core_id"] == tick["acting_core_id"], "attention view belongs to another core")
    require(record["attention_view"]["profile_id"] == record["authority"]["profile_id"], "attention/authority profile mismatch")
    if record["authority"]["profile_id"] == FOCUS_PROFILE:
        for name in REGIONS:
            require(region_text(snapshot, name), f"focus fixture leaves {name} empty")
            require(
                record["attention_view"]["regions"][name]["active_intervals"],
                f"focus fixture does not expose active text from {name}",
            )

    active_stream = reconstruct_active_stream(record)
    base_namespace = record["coverage_manifest"]["base_field"]
    require(base_namespace["source_digest"] == digest_text(active_stream), "base coverage source is not the acting core's active view")
    require(base_namespace["expected_characters"] == len(active_stream), "base active character count mismatch")
    require(record["difficulty"]["active_character_count"] == len(active_stream), "difficulty active count mismatch")
    require(record["difficulty"]["base_field_page_count"] == len(base_namespace["expected_page_indices"]), "difficulty page count mismatch")

    validate_sibling_set(record)
    sibling_namespace = record["coverage_manifest"]["sibling_delta_set"]
    sibling_text = canonical_json(record["sibling_delta_set"]) if tick["phase"] != "proposal" else ""
    require(sibling_namespace["source_digest"] == digest_text(sibling_text), "sibling coverage source digest mismatch")
    require(sibling_namespace["expected_characters"] == len(sibling_text), "sibling coverage length mismatch")

    observed = independent_coverage_oracle(record["coverage_manifest"])
    expected_failure = record["expected"]["failure_code"]
    require(observed == expected_failure, f"independent coverage oracle expected {expected_failure!r}, observed {observed!r}")
    require(record["coverage_manifest"]["observed_failure_code"] == observed, "builder and independent coverage oracles disagree")
    require(record["builder_validation"]["coverage_oracle_expected"] == expected_failure, "builder expected failure claim mismatch")
    require(record["builder_validation"]["coverage_oracle_observed"] == observed, "builder observed failure claim mismatch")
    require(record["builder_validation"]["external_validation"] is None, "builder claims external validation")
    require(record["expected"]["finalization_allowed"] == (observed is None), "finalization flag differs from coverage result")

    validate_transport(record)
    validate_authority(record, profiles)

    for evidence in record["evidence_spans"]:
        source = region_text(snapshot, evidence["region"])
        require(source[evidence["start"]:evidence["end"]] == evidence["exact_text"], "evidence pointer is not exact")
    soul = record["soul_transition"]
    require(soul["inhale_count"] == soul["exhale_count"] == 1, "soul phase boundary count mismatch")
    require(soul["authoritative_exact_memory"] is False, "soul is asserted as exact authority")
    if record["phase_result"]["outcome"] == "delta":
        envelope = record["phase_result"]["delta_envelope"]
        require(envelope["soul_before"] == soul["before"], "target envelope soul-before mismatch")
        require(envelope["soul_after"] == soul["after"], "target envelope soul-after mismatch")

    texts = [region_text(snapshot, name) for name in REGIONS]
    texts.extend(envelope["payload"]["exact_text"] for envelope in record["sibling_delta_set"]["envelopes"])
    unsupported = sorted({char for text in texts for char in text if char not in alphabet_chars})
    require(not unsupported, f"unsupported alphabet characters: {unsupported!r}")
    require(record["alphabet"]["supported"] is True, "builder quarantined a supposedly accepted record")


def validate_record_collection(
    records: list[dict[str, Any]],
    profiles: dict[str, Any],
    alphabet_chars: set[str],
    *,
    require_complete_groups: bool,
) -> None:
    require(records, "record collection is empty")
    ids = [record["example_id"] for record in records]
    require(len(ids) == len(set(ids)), "duplicate example ids")
    for record in records:
        validate_record(record, profiles, alphabet_chars)
    by_group: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_group[record["counterfactual_group"]].add(record["counterfactual_variant"])
        group_splits[record["counterfactual_group"]].add(record["split"])
    require(all(len(values) == 1 for values in group_splits.values()), "counterfactual lineage crossed splits")
    if require_complete_groups:
        require(all(values == REQUIRED_VARIANTS for values in by_group.values()), "incomplete counterfactual variant group")


def validate_starter(package_root: Path, profiles: dict[str, Any], alphabet_chars: set[str]) -> dict[str, Any]:
    records = load_jsonl(package_root / "starter_examples.jsonl")
    validate_record_collection(records, profiles, alphabet_chars, require_complete_groups=False)
    require(set(record["family"] for record in records) == set("ABCDEFGHIJKLM"), "starter families A-M are incomplete")
    council = [record for record in records if record["family"] == "J" and record["tick"]["tick_seq"] == 777]
    require(len(council) == 3, "starter council episode does not contain three phases")
    require([(record["tick"]["phase"], record["tick"]["phase_seq"]) for record in council] == list(PHASE_TO_SEQ.items()), "starter council phases are not 0/1/2 in one tick")
    return {"records": len(records), "digest": sha256_bytes((package_root / "starter_examples.jsonl").read_bytes())}


def validate_frozen(package_root: Path, profiles: dict[str, Any], alphabet_chars: set[str]) -> dict[str, Any]:
    records = load_jsonl(package_root / "frozen_evals.jsonl")
    validate_record_collection(records, profiles, alphabet_chars, require_complete_groups=True)
    correct = [record for record in records if record["counterfactual_variant"] == "correct"]
    require({record["difficulty"]["base_field_page_count"] for record in correct} == REQUIRED_STRATA, "frozen eval lacks exact 1/2/4/8 strata")
    return {"records": len(records), "digest": sha256_bytes((package_root / "frozen_evals.jsonl").read_bytes())}


def validate_generated(
    generated_root: Path,
    profiles: dict[str, Any],
    alphabet_chars: set[str],
) -> dict[str, Any]:
    manifest = load_json(generated_root / "manifest.json")
    require(manifest["schema"] == "axon-complete-field-curriculum-manifest-v0.2", "wrong generated manifest schema")
    require(manifest["external_validation"] is None, "builder manifest claims external validation")
    require(manifest["builder_validation"]["schema_validation"] is None, "builder claims JSON Schema validation")
    require(manifest["builder_validation"]["deterministic_replay"] is None, "builder claims two-run replay")
    records = []
    for split in ("train", "dev", "test"):
        path = generated_root / manifest["split_files"][split]["path"]
        require(sha256_bytes(path.read_bytes()) == manifest["split_files"][split]["digest"], f"{split} file hash mismatch")
        split_records = load_jsonl(path)
        require(len(split_records) == manifest["split_files"][split]["records"], f"{split} record count mismatch")
        require(all(record["split"] == split for record in split_records), f"{split} file contains wrong split")
        records.extend(split_records)
    validate_record_collection(records, profiles, alphabet_chars, require_complete_groups=True)
    correct = [record for record in records if record["counterfactual_variant"] == "correct"]
    page_counts = Counter(record["difficulty"]["base_field_page_count"] for record in correct)
    bin_counts = Counter(record["difficulty"]["evidence_position"] for record in correct)
    require(set(page_counts) == REQUIRED_STRATA and all(page_counts[value] for value in REQUIRED_STRATA), "generated page strata incomplete")
    require(set(bin_counts) == REQUIRED_BINS and all(bin_counts[value] for value in REQUIRED_BINS), "generated evidence positions incomplete")
    require(manifest["record_count"] == len(records), "generated manifest record count mismatch")
    require(manifest["group_count"] * len(REQUIRED_VARIANTS) == len(records), "group count does not bind complete variants")
    require({int(key): value for key, value in manifest["page_strata"].items()} == dict(page_counts), "manifest page distribution mismatch")
    require(manifest["evidence_position_counts"] == {key: bin_counts[key] for key in ("first", "early", "middle", "late", "last")}, "manifest evidence distribution mismatch")
    return {
        "records": len(records),
        "groups": manifest["group_count"],
        "page_strata": dict(sorted(page_counts.items())),
        "evidence_positions": dict(sorted(bin_counts.items())),
        "manifest_digest": sha256_bytes((generated_root / "manifest.json").read_bytes()),
    }


def compare_directories(left: Path, right: Path) -> dict[str, str]:
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    require(left_files == right_files, "two clean runs produced different file lists")
    hashes = {}
    for relative in left_files:
        left_bytes = (left / relative).read_bytes()
        right_bytes = (right / relative).read_bytes()
        require(left_bytes == right_bytes, f"two clean runs differ at {relative}")
        hashes[str(relative).replace("\\", "/")] = sha256_bytes(left_bytes)
    return hashes


def run_replay(package_root: Path, groups: int, page_size: int) -> tuple[dict[str, str], Path, Path, tempfile.TemporaryDirectory[str]]:
    temp = tempfile.TemporaryDirectory(prefix="axon-v02-replay-")
    root = Path(temp.name)
    left = root / "run-a"
    right = root / "run-b"
    command_base = [
        sys.executable,
        str(package_root / "generate_curriculum.py"),
        "--package-root",
        str(package_root),
        "--groups",
        str(groups),
        "--page-size",
        str(page_size),
    ]
    for destination in (left, right):
        completed = subprocess.run(
            [*command_base, "--output-root", str(destination)],
            cwd=package_root,
            text=True,
            capture_output=True,
            check=False,
        )
        require(completed.returncode == 0, f"builder replay failed: {completed.stderr or completed.stdout}")
    return compare_directories(left, right), left, right, temp


def package_file_hashes(package_root: Path) -> dict[str, str]:
    excluded = {"chatgpt_verification_evidence.json", "PACKAGE_MANIFEST.json"}
    result = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name in excluded or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(package_root)).replace("\\", "/")
        result[relative] = sha256_bytes(path.read_bytes())
    return result


def validate_package_manifest(package_root: Path) -> dict[str, Any]:
    manifest = load_json(package_root / "PACKAGE_MANIFEST.json")
    require(
        manifest["schema"] == "axon-inspection-package-manifest-v0.2",
        "wrong package manifest schema",
    )
    require(manifest["status"] == "REVIEW_ONLY", "package manifest is not review-only")
    expected = package_file_hashes(package_root)
    listed = {entry["path"]: entry for entry in manifest["files"]}
    require(set(listed) == set(expected), "package manifest file list mismatch")
    require(manifest["file_count"] == len(expected), "package manifest file count mismatch")
    for path, digest in expected.items():
        entry = listed[path]
        require(entry["algorithm"] == "sha256", f"{path} uses a non-SHA256 package digest")
        require(entry["digest"] == digest, f"{path} package digest mismatch")
        require(entry["size"] == (package_root / path).stat().st_size, f"{path} package size mismatch")
    return {
        "files": len(expected),
        "digest": sha256_bytes((package_root / "PACKAGE_MANIFEST.json").read_bytes()),
    }


def verify(
    package_root: Path,
    *,
    generated_root: Path | None,
    replay_groups: int,
    replay_page_size: int,
) -> dict[str, Any]:
    for name in (
        "complete_field_example.schema.json",
        "curriculum_manifest.schema.json",
        "authority_profiles.schema.json",
    ):
        schema = load_json(package_root / name)
        require(schema["$schema"] == "https://json-schema.org/draft/2020-12/schema", f"{name} is not draft 2020-12")
    profiles = load_json(package_root / "authority_profiles.json")
    alphabet = load_json(package_root / "alphabet_manifest.json")
    package_manifest = validate_package_manifest(package_root)
    validate_profiles(profiles)
    alphabet_chars = validate_alphabet(alphabet)
    starter = validate_starter(package_root, profiles, alphabet_chars)
    frozen = validate_frozen(package_root, profiles, alphabet_chars)

    generated = None
    if generated_root is not None:
        generated = validate_generated(generated_root, profiles, alphabet_chars)

    replay_hashes, replay_root, _, temp = run_replay(
        package_root, replay_groups, replay_page_size
    )
    try:
        replay_validation = validate_generated(replay_root, profiles, alphabet_chars)
    finally:
        temp.cleanup()
    return {
        "schema": "axon-curriculum-verification-evidence-v0.2",
        "verifier_version": VERIFIER_VERSION,
        "stamp": "ChatGPT / GPT-5 / 2026-08-18",
        "run_id": datetime.now(timezone.utc).strftime("verify-%Y%m%dT%H%M%S%fZ"),
        "status": "PASS",
        "checks": {
            "schema_documents_parse": "PASS",
            "package_manifest": "PASS",
            "authority_profiles": "PASS",
            "immutable_evidence_content": "PASS",
            "independent_attention_switches": "PASS",
            "focus_all_ten_regions_attended": "PASS",
            "focus_three_region_delta": "PASS",
            "alphabet_manifest": "PASS",
            "snapshot_identity": "PASS",
            "sibling_identity_and_roster": "PASS",
            "base_and_sibling_coverage": "PASS",
            "one_tick_three_phases": "PASS",
            "soul_phase_bindings": "PASS",
            "atomic_chunk_assembly": "PASS",
            "counterfactual_completeness": "PASS",
            "page_strata_1_2_4_8": "PASS",
            "eligible_evidence_positions": "PASS",
            "two_clean_run_byte_replay": "PASS",
        },
        "starter": starter,
        "frozen": frozen,
        "package_manifest": package_manifest,
        "provided_generated_root": generated,
        "replay": {"files": replay_hashes, "validation": replay_validation},
        "package_files": package_file_hashes(package_root),
        "failures": [],
        "training_launched": False,
        "runtime_modified": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--replay-groups", type=int, default=20)
    parser.add_argument("--replay-page-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = verify(
        args.package_root.resolve(),
        generated_root=args.generated_root.resolve() if args.generated_root else None,
        replay_groups=args.replay_groups,
        replay_page_size=args.replay_page_size,
    )
    rendered = json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.evidence_out:
        args.evidence_out.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
