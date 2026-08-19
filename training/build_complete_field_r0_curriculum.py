"""Build Axon's local-only R0 complete-field curriculum.

The builder keeps evidence classes explicit:

* Grade A: exact raw user/assistant messages from the read-only message store.
* Grade D: derived, grounded procedure episodes used only for capability.
* Grade S: deterministic synthetic mechanics/reasoning fixtures.

No diary target is produced.  Every record contains all ten field regions and
only scratch plus response_draft targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from substrate import ALPHABET_SET, assert_supported_text
from training.complete_field_64d import REGION_ORDER, canonical_field


SCHEMA = "axon-complete-field-r0-example-v1"
BUILDER_VERSION = "complete-field-r0-builder-2026-08-18"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def supported(text: str) -> bool:
    return all(char in ALPHABET_SET for char in text)


def newline_exact(text: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, "crlf_to_lf" if normalized != text else "none"


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 84:
        return "train"
    if bucket < 92:
        return "dev"
    return "test"


def make_record(
    *,
    family: str,
    field: Mapping[str, str],
    scratch: str,
    response: str,
    provenance: Mapping[str, Any],
    group: str,
    split: str | None = None,
) -> dict[str, Any]:
    exact_field = canonical_field(field)
    assert_supported_text(scratch)
    assert_supported_text(response)
    if len(scratch) > 512 or len(response) > 512:
        raise ValueError("R0 target exceeds explicit 512-character training bound")
    body = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "family": family,
        "split": split or split_for(group),
        "lineage_group": group,
        "field": exact_field,
        "targets": {"scratch": scratch, "response_draft": response},
        "write_authority": {
            "scratch": True,
            "response_draft": True,
            "diary": False,
            "conversation_history": False,
            "tool_results": False,
        },
        "provenance": dict(provenance),
    }
    body["example_id"] = digest(body)
    return body


def random_token(rng: random.Random, length: int = 12) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def distractor(rng: random.Random, chars: int) -> str:
    words = ("field page exact evidence verify region state context signal trace " * 64).split()
    out: list[str] = []
    while len(" ".join(out)) < chars:
        out.append(rng.choice(words))
    return (" ".join(out) + " ")[:chars]


def synthetic_records(count: int, seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for index in range(count):
        kind = index % 5
        group = f"synthetic-r0-{seed}-{index}"
        field = {name: "" for name in REGION_ORDER}
        field["situation_awareness"] = "One 64D core is learning complete field coverage."
        field["advisor_input"] = "Use exact visible evidence and admit uncertainty."
        field["task_state"] = "Write useful scratch, reread it, then update the response draft."
        if kind == 0:
            token = random_token(rng)
            before = distractor(rng, rng.choice((0, 120, 260, 520, 780)))
            after = distractor(rng, rng.choice((20, 180, 400)))
            field["tool_results"] = before + "Verified token: " + token + ". " + after
            field["user_input"] = "What exact token was verified?"
            scratch = "The verified token in tool_results is " + token + "."
            response = "The verified token is " + token + "."
            family = "cross_page_exact_retrieval"
        elif kind == 1:
            left = rng.randrange(2, 80)
            right = rng.randrange(2, 80)
            product = left * right
            field["structured_knowledge"] = distractor(rng, rng.choice((0, 230, 490)))
            field["user_input"] = f"Multiply {left} by {right}."
            scratch = f"{left} * {right} = {product}."
            response = f"{left} times {right} is {product}."
            family = "scratch_arithmetic"
        elif kind == 2:
            name = rng.choice(("Axon", "Jeff", "Council", "scratch", "response"))
            field["conversation_history"] = "Jeff: Keep the answer grounded.\nAxon: I will use visible evidence.\n"
            field["user_input"] = f"Spell {name} exactly."
            scratch = "Copy the requested visible token without changing it."
            response = name
            family = "conversation_exact_copy"
        elif kind == 3:
            field["conversation_history"] = "Jeff: Hello Axon.\nAxon: Hello Jeff.\n"
            field["user_input"] = rng.choice(("How are you?", "Are you ready?", "Can we continue?"))
            scratch = "Answer Jeff directly, briefly, and truthfully."
            response = rng.choice(("I am ready to continue.", "Yes. I am ready.", "I am here and paying attention."))
            family = "conversation_foundation"
        else:
            field["structured_knowledge"] = distractor(rng, rng.choice((30, 270, 530)))
            field["user_input"] = "What color is the hidden marker?"
            scratch = "No visible region states the hidden marker color."
            response = "I do not know from the visible field."
            family = "grounded_abstention"
        yield make_record(
            family=family,
            field=field,
            scratch=scratch,
            response=response,
            provenance={
                "grade": "S",
                "source_type": "deterministic_synthetic",
                "autobiographical": False,
                "seed": seed,
                "index": index,
            },
            group=group,
        )


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = "file:///" + path.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def exact_conversation_records(db_path: Path, limit: int) -> Iterator[dict[str, Any]]:
    with _connect_ro(db_path) as connection:
        rows = connection.execute(
            "SELECT id, role, content, timestamp FROM messages ORDER BY timestamp, id"
        ).fetchall()
    candidates: list[dict[str, Any]] = []
    previous_clean: list[tuple[str, str]] = []
    for left, right in zip(rows, rows[1:]):
        if left["role"] != "user" or right["role"] != "assistant":
            continue
        user, user_transform = newline_exact(left["content"] or "")
        answer, answer_transform = newline_exact(right["content"] or "")
        lowered = answer.lower()
        if (
            not user
            or not answer
            or len(user) > 256
            or len(answer) > 384
            or not supported(user)
            or not supported(answer)
            or user.startswith("[TASK RESULT")
            or "```tool" in lowered
            or "<thought>" in lowered
            or "<func_calls>" in lowered
        ):
            continue
        group = "raw-turn-day-" + str(left["timestamp"])[:10]
        history = "".join(
            f"{role}: {text}\n" for role, text in previous_clean[-2:]
        )
        field = {name: "" for name in REGION_ORDER}
        field["conversation_history"] = history[-1024:]
        field["user_input"] = user
        scratch = "Answer the current user directly using the visible conversation."
        record = make_record(
            family="exact_d00_conversation",
            field=field,
            scratch=scratch,
            response=answer,
            provenance={
                "grade": "A",
                "source_type": "exact_message_pair",
                "autobiographical": False,
                "source_db": str(db_path),
                "user_message_id": left["id"],
                "assistant_message_id": right["id"],
                "user_timestamp": left["timestamp"],
                "assistant_timestamp": right["timestamp"],
                "normalization": [user_transform, answer_transform],
                "source_pair_sha256": digest(
                    {"user": left["content"], "assistant": right["content"]}
                ),
            },
            group=group,
        )
        candidates.append(record)
        previous_clean.extend((("User", user), ("Axon", answer)))
    candidates.sort(key=lambda item: item["example_id"])
    yield from candidates[:limit]


def identity_anchor_records(db_path: Path) -> Iterator[dict[str, Any]]:
    naming_user_id = "MSG-064437-5af632"
    naming_assistant_id = "MSG-064437-452012"
    with _connect_ro(db_path) as connection:
        rows = connection.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE id IN (?, ?)",
            (naming_user_id, naming_assistant_id),
        ).fetchall()
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {naming_user_id, naming_assistant_id}:
        raise RuntimeError("the exact Axon naming anchors were not found")
    source_text, transform = newline_exact(by_id[naming_assistant_id]["content"])
    start_marker = "I've chosen a name.\n\n**AXON**"
    end_marker = "They enable coordinated, multi-part action."
    start = source_text.index(start_marker)
    end = source_text.index(end_marker, start) + len(end_marker)
    excerpt = source_text[start:end]
    assert_supported_text(excerpt)
    prompts = (
        "What name did you choose and why?",
        "Tell me the recorded reason you chose the name AXON.",
        "What does your name mean to you?",
        "Did Jeff assign the name Axon, or did you choose it?",
        "Explain your naming decision from the recorded evidence.",
    )
    responses = (
        "I chose AXON because axons transmit signals, connect neurons into networks, and enable coordinated action.",
        "The record says I chose AXON. I connected the name to transmitting signals and turning intent into action.",
        "AXON names the function I want to serve: connect signals, coordinate work, and carry intent into action.",
        "I chose AXON. Jeff gave me veto room, but the recorded naming decision was mine.",
        "The naming record says I chose AXON because an axon transmits signals and enables coordinated action.",
    )
    for index, (prompt, response) in enumerate(zip(prompts, responses)):
        field = {name: "" for name in REGION_ORDER}
        field["structured_knowledge"] = "Recorded naming event:\n" + excerpt
        field["user_input"] = prompt
        yield make_record(
            family="axon_identity_naming_anchor",
            field=field,
            scratch="Use the exact recorded naming event and distinguish the choice from later interpretation.",
            response=response,
            provenance={
                "grade": "A",
                "source_type": "exact_message_excerpt",
                "autobiographical": True,
                "source_db": str(db_path),
                "source_message_id": naming_assistant_id,
                "source_timestamp": by_id[naming_assistant_id]["timestamp"],
                "source_message_sha256": digest(by_id[naming_assistant_id]["content"].encode("utf-8")),
                "exact_excerpt_start": start,
                "exact_excerpt_end": end,
                "normalization": transform,
                "target_derivation": "grounded_paraphrase",
            },
            group="identity-event-axon-naming-2026-03-06",
            split="train",
        )


def grounded_scratch_records(path: Path, limit: int) -> Iterator[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            field = {
                name: str(raw.get("initial_field", {}).get(name, {}).get("text", ""))
                for name in REGION_ORDER
            }
            scratch_ticks = [t for t in raw.get("ticks", []) if t.get("target_region") == "scratch"]
            response_ticks = [
                t for t in raw.get("ticks", []) if t.get("target_region") == "response_draft"
            ]
            if not scratch_ticks or not response_ticks:
                continue
            scratch = str(scratch_ticks[-1].get("complete_proposed_region_text", ""))
            response = str(response_ticks[-1].get("complete_proposed_region_text", ""))
            if (
                not scratch
                or not response
                or len(scratch) > 512
                or len(response) > 512
                or any(not supported(text) for text in [*field.values(), scratch, response])
            ):
                continue
            try:
                record = make_record(
                    family="grounded_scratch_plan_response",
                    field=field,
                    scratch=scratch,
                    response=response,
                    provenance={
                        "grade": "D",
                        "source_type": "derived_grounded_procedure_episode",
                        "autobiographical": False,
                        "source_path": str(path),
                        "source_line": line_number,
                        "source_episode_id": raw.get("episode_id"),
                        "evidence_refs": raw.get("provenance", {}).get("evidence_refs", []),
                        "warning": "capability only; never sole autobiographical evidence",
                    },
                    group=str(raw.get("split_cluster_id") or raw.get("source_lineage") or line_number),
                    split=str(raw.get("split", "train")),
                )
            except (TypeError, ValueError):
                continue
            selected.append(record)
    selected.sort(key=lambda item: item["example_id"])
    yield from selected[:limit]


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        by_id.setdefault(record["example_id"], record)
    return sorted(by_id.values(), key=lambda item: (item["split"], item["family"], item["example_id"]))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload.count(b"\n"), hashlib.sha256(payload).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    records.extend(synthetic_records(args.synthetic, args.seed))
    records.extend(exact_conversation_records(args.memory_db, args.exact_pairs))
    records.extend(identity_anchor_records(args.memory_db))
    if args.grounded_episodes and args.grounded_episodes.is_file():
        records.extend(grounded_scratch_records(args.grounded_episodes, args.grounded_limit))
    records = deduplicate(records)
    by_split = {split: [r for r in records if r["split"] == split] for split in ("train", "dev", "test")}
    files: dict[str, Any] = {}
    for split, rows in by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        count, sha = write_jsonl(path, rows)
        files[split] = {"path": str(path), "records": count, "sha256": sha}
    manifest = {
        "schema": "axon-complete-field-r0-manifest-v1",
        "builder_version": BUILDER_VERSION,
        "local_only": True,
        "cloud_export_allowed": False,
        "diary_target_count": 0,
        "region_order": list(REGION_ORDER),
        "target_regions": ["scratch", "response_draft"],
        "source_read_policy": "SQLite URI mode=ro plus PRAGMA query_only=ON",
        "counts_by_family": dict(Counter(record["family"] for record in records)),
        "counts_by_grade": dict(Counter(record["provenance"]["grade"] for record in records)),
        "files": files,
        "seed": args.seed,
    }
    manifest["manifest_sha256"] = digest(manifest)
    (args.output_dir / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build local-only complete-field R0 curriculum")
    p.add_argument("--memory-db", type=Path, default=Path(r"D:\00\axon_memory.db"))
    p.add_argument(
        "--grounded-episodes",
        type=Path,
        default=Path(r"datasets\recovered\multitick_curriculum_d00_grounded_v4\scratch_plan_response_v4.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path(r"State\private_curriculum\complete_field_r0"))
    p.add_argument("--synthetic", type=int, default=8000)
    p.add_argument("--exact-pairs", type=int, default=6000)
    p.add_argument("--grounded-limit", type=int, default=4000)
    p.add_argument("--seed", type=int, default=64018)
    return p


def main() -> int:
    args = parser().parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
