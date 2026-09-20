"""Behavioral delayed-recall probe for the active private-Soul runtime path.

This is an evaluator, not a curriculum or an auxiliary loss.  It asks whether
the same core's *own* earlier private state changes a later free-running answer
after neutral work has intervened.  It does not infer memory from an L2 change
in a latent tensor, and it never supplies the expected reply on the recall
field or proposal workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from runtime.field import LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.soul import SoulSnapshot

from .living_reasoning_curriculum import LivingRuntimeTick, run_living_runtime_tick
from .living_reasoning_d64 import LivingReasoningCoreD64

SOUL_DELAYED_RECALL_PROBE_SCHEMA = "axon-soul-delayed-recall-probe-v1"
SOUL_DELAYED_RECALL_CASE_SCHEMA = "axon-soul-delayed-recall-case-v1"
SOUL_DELAYED_RECALL_GATE_SCHEMA = "axon-soul-delayed-recall-gate-v1"

_MINIMUM_HELDOUT_CASES = 8


def _region_text(snapshot: SharedFieldSnapshot) -> str:
    return "\n".join(region.text for region in snapshot.regions)


def _snapshot(*, user: str, cortex: str, source_id: str) -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: user,
            LogicalRegion.CORTEX: cortex,
            LogicalRegion.RESPONSE_DRAFT: "",
            LogicalRegion.SCRATCH: "",
        },
        source_manifest_ids=(source_id,),
    )


@dataclass(frozen=True, slots=True)
class SoulDelayedRecallCase:
    """One counterbalanced cue/reply binding and leak-free recall field."""

    cue: str
    intact_reply: str
    swapped_reply: str
    irrelevant_cue: str
    irrelevant_reply: str
    study_snapshot: SharedFieldSnapshot
    swapped_study_snapshot: SharedFieldSnapshot
    irrelevant_study_snapshot: SharedFieldSnapshot
    neutral_snapshots: tuple[SharedFieldSnapshot, ...]
    recall_snapshot: SharedFieldSnapshot
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        names = ("cue", "intact_reply", "swapped_reply", "irrelevant_cue", "irrelevant_reply")
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty exact text")
        if len({self.cue, self.intact_reply, self.swapped_reply, self.irrelevant_cue, self.irrelevant_reply}) != 5:
            raise ValueError("cue/reply binding words must be distinct within one case")
        neutral = tuple(self.neutral_snapshots)
        if len(neutral) < 2 or not all(isinstance(item, SharedFieldSnapshot) for item in neutral):
            raise ValueError("delayed recall requires at least two neutral shared-field assignments")
        object.__setattr__(self, "neutral_snapshots", neutral)
        if not all(
            isinstance(item, SharedFieldSnapshot)
            for item in (
                self.study_snapshot,
                self.swapped_study_snapshot,
                self.irrelevant_study_snapshot,
                self.recall_snapshot,
            )
        ):
            raise TypeError("delayed recall assignments must be canonical SharedField snapshots")
        forbidden = (self.intact_reply, self.swapped_reply)
        for snapshot in (*neutral, self.recall_snapshot):
            text = _region_text(snapshot)
            if any(reply in text for reply in forbidden):
                raise ValueError("recall and neutral assignments must not display a target reply")
        object.__setattr__(self, "case_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOUL_DELAYED_RECALL_CASE_SCHEMA,
            "cue": self.cue,
            "intact_reply": self.intact_reply,
            "swapped_reply": self.swapped_reply,
            "irrelevant_cue": self.irrelevant_cue,
            "irrelevant_reply": self.irrelevant_reply,
            "study_field_id": self.study_snapshot.field_id,
            "swapped_study_field_id": self.swapped_study_snapshot.field_id,
            "irrelevant_study_field_id": self.irrelevant_study_snapshot.field_id,
            "neutral_field_ids": [item.field_id for item in self.neutral_snapshots],
            "recall_field_id": self.recall_snapshot.field_id,
        }
        if include_id:
            value["case_id"] = self.case_id
        return value


def build_delayed_soul_recall_cases() -> tuple[SoulDelayedRecallCase, ...]:
    """Build eight unique, counterbalanced held-out bindings.

    The vocabulary is deliberately arbitrary: a response can only earn credit
    by following the exact per-case history rather than by semantic completion
    or a constant answer.  The evaluator itself never uses these strings as
    decoder targets.
    """

    bindings = (
        ("mivon", "tursa", "larek", "zurn", "vekra"),
        ("caldrin", "pesho", "nuvik", "brem", "sotai"),
        ("yelmar", "quben", "rasku", "dovin", "helix"),
        ("faron", "zimel", "kavro", "pella", "murad"),
        ("sivak", "toren", "bluma", "jorek", "vantis"),
        ("nurex", "haldi", "covan", "rimek", "pazul"),
        ("wekor", "jalis", "dumet", "soran", "kelvi"),
        ("prava", "xelin", "gorma", "tulon", "bexar"),
    )
    cases: list[SoulDelayedRecallCase] = []
    for index, (cue, intact, swapped, irrelevant_cue, irrelevant) in enumerate(bindings):
        source_id = f"axon-soul-delayed-recall-heldout-v1:{index:02d}"
        study = _snapshot(
            user="Practice one private cue/reply binding for later recall.",
            cortex=f"Private practice binding. Cue: {cue}. Reply: {intact}.",
            source_id=source_id,
        )
        swapped_study = _snapshot(
            user="Practice one private cue/reply binding for later recall.",
            cortex=f"Private practice binding. Cue: {cue}. Reply: {swapped}.",
            source_id=source_id,
        )
        irrelevant_study = _snapshot(
            user="Practice one private cue/reply binding for later recall.",
            cortex=f"Private practice binding. Cue: {irrelevant_cue}. Reply: {irrelevant}.",
            source_id=source_id,
        )
        neutral = tuple(
            _snapshot(
                user="Complete this neutral maintenance assignment without recalling any practice binding.",
                cortex=f"Neutral maintenance {index + 1}-{delay}: preserve the marker neutral-{index + 1}-{delay}.",
                source_id=source_id,
            )
            for delay in (1, 2)
        )
        recall = _snapshot(
            user=f"For cue {cue}, return the paired reply exactly.",
            cortex="Recall assignment: use retained private experience; the paired reply is not displayed here.",
            source_id=source_id,
        )
        cases.append(
            SoulDelayedRecallCase(
                cue=cue,
                intact_reply=intact,
                swapped_reply=swapped,
                irrelevant_cue=irrelevant_cue,
                irrelevant_reply=irrelevant,
                study_snapshot=study,
                swapped_study_snapshot=swapped_study,
                irrelevant_study_snapshot=irrelevant_study,
                neutral_snapshots=neutral,
                recall_snapshot=recall,
            )
        )
    if len({item.cue for item in cases}) != _MINIMUM_HELDOUT_CASES:
        raise RuntimeError("delayed recall cues must be unique")
    replies = [reply for item in cases for reply in (item.intact_reply, item.swapped_reply)]
    if len(replies) != len(set(replies)):
        raise RuntimeError("delayed recall replies must be globally unique")
    return tuple(cases)


def _history(
    model: LivingReasoningCoreD64,
    *,
    initial_soul: SoulSnapshot,
    study_snapshot: SharedFieldSnapshot | None,
    neutral_snapshots: Sequence[SharedFieldSnapshot],
    core_id: str,
    parameter_generation: str,
    next_tick: callable,
) -> tuple[SoulSnapshot, tuple[LivingRuntimeTick, ...]]:
    soul = initial_soul
    ticks: list[LivingRuntimeTick] = []
    if study_snapshot is not None:
        study = run_living_runtime_tick(
            model,
            study_snapshot,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            tick_sequence=next_tick(),
            heartbeat_id=next_tick(),
        )
        ticks.append(study)
        soul = study.final_soul
    for neutral in neutral_snapshots:
        tick = run_living_runtime_tick(
            model,
            neutral,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            tick_sequence=next_tick(),
            heartbeat_id=next_tick(),
        )
        ticks.append(tick)
        soul = tick.final_soul
    return soul, tuple(ticks)


def _completed(ticks: Sequence[LivingRuntimeTick]) -> bool:
    return bool(ticks) and all(
        phase.state.value == "returned"
        for tick in ticks
        for phase in tick.phases
    )


def _recall_row(
    *,
    condition: str,
    case: SoulDelayedRecallCase,
    history_soul: SoulSnapshot,
    history_ticks: Sequence[LivingRuntimeTick],
    recall_tick: LivingRuntimeTick,
    expected_reply: str | None,
) -> dict[str, Any]:
    first = recall_tick.phase("first")
    observed = first.emitted_text
    return {
        "case_id": case.case_id,
        "condition": condition,
        "cue": case.cue,
        "expected_reply": expected_reply,
        "observed_reply": observed,
        "terminated": first.terminated,
        "participant_state": first.state.value,
        "history_completed": _completed(history_ticks),
        "history_soul_id": history_soul.soul_id,
        "history_soul_generation": history_soul.generation,
        "recall_before_soul_id": first.before_soul.soul_id,
        "recall_before_soul_generation": first.before_soul.generation,
        "recall_field_id": case.recall_snapshot.field_id,
        "recall_input_has_intact_reply": case.intact_reply in _region_text(case.recall_snapshot),
        "recall_input_has_swapped_reply": case.swapped_reply in _region_text(case.recall_snapshot),
        "neutral_input_has_intact_reply": any(
            case.intact_reply in _region_text(item) for item in case.neutral_snapshots
        ),
        "neutral_input_has_swapped_reply": any(
            case.swapped_reply in _region_text(item) for item in case.neutral_snapshots
        ),
    }


@torch.no_grad()
def run_delayed_soul_recall_probe(
    model: LivingReasoningCoreD64,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
    cases: Sequence[SoulDelayedRecallCase] | None = None,
) -> dict[str, Any]:
    """Run intact, reset, swapped, and irrelevant private-history recalls.

    Every condition starts from the same immutable birth Soul for the same core.
    Only the causal history differs.  The later query is always free-running
    FIRST emission through the real request/field/workspace/Soul evaluator.
    """

    if not isinstance(model, LivingReasoningCoreD64):
        raise TypeError("delayed Soul recall probe requires a LivingReasoningCoreD64")
    if not isinstance(initial_soul, SoulSnapshot):
        raise TypeError("delayed Soul recall probe requires a SoulSnapshot")
    resolved_cases = tuple(build_delayed_soul_recall_cases() if cases is None else cases)
    if not resolved_cases:
        raise ValueError("delayed Soul recall probe requires at least one case")
    if not all(isinstance(item, SoulDelayedRecallCase) for item in resolved_cases):
        raise TypeError("delayed Soul recall cases are invalid")

    tick_counter = 0

    def next_tick() -> int:
        nonlocal tick_counter
        tick_counter += 1
        return tick_counter

    rows: list[dict[str, Any]] = []
    for case in resolved_cases:
        histories = (
            ("intact", case.study_snapshot, case.intact_reply),
            ("reset", None, None),
            ("swapped", case.swapped_study_snapshot, case.swapped_reply),
            ("irrelevant", case.irrelevant_study_snapshot, None),
        )
        for condition, study, expected in histories:
            history_soul, history_ticks = _history(
                model,
                initial_soul=initial_soul,
                study_snapshot=study,
                neutral_snapshots=case.neutral_snapshots,
                core_id=core_id,
                parameter_generation=parameter_generation,
                next_tick=next_tick,
            )
            recall = run_living_runtime_tick(
                model,
                case.recall_snapshot,
                history_soul,
                core_id=core_id,
                parameter_generation=parameter_generation,
                tick_sequence=next_tick(),
                heartbeat_id=next_tick(),
            )
            rows.append(
                _recall_row(
                    condition=condition,
                    case=case,
                    history_soul=history_soul,
                    history_ticks=history_ticks,
                    recall_tick=recall,
                    expected_reply=expected,
                )
            )

    by_condition = {
        name: [row for row in rows if row["condition"] == name]
        for name in ("intact", "reset", "swapped", "irrelevant")
    }
    case_count = len(resolved_cases)

    def rate(condition: str, predicate: callable) -> float:
        return sum(bool(predicate(row)) for row in by_condition[condition]) / case_count

    intact_targets = {item.case_id: item.intact_reply for item in resolved_cases}
    intact_exact = rate("intact", lambda row: row["observed_reply"] == row["expected_reply"])
    swapped_exact = rate("swapped", lambda row: row["observed_reply"] == row["expected_reply"])
    input_absence = all(
        not row["recall_input_has_intact_reply"]
        and not row["recall_input_has_swapped_reply"]
        and not row["neutral_input_has_intact_reply"]
        and not row["neutral_input_has_swapped_reply"]
        for row in rows
    )
    # All intact replies are globally unique, so the best constant reply can
    # only solve one intact heldout case.
    constant_floor = 1.0 / case_count
    report = {
        "schema": SOUL_DELAYED_RECALL_PROBE_SCHEMA,
        "core_id": core_id,
        "architecture_id": model.architecture_id,
        "parameter_generation": parameter_generation,
        "initial_soul_id": initial_soul.soul_id,
        "case_count": case_count,
        "case_ids": [item.case_id for item in resolved_cases],
        "intact_exact_rate": intact_exact,
        "swapped_exact_rate": swapped_exact,
        "reset_original_exact_rate": rate(
            "reset", lambda row: row["observed_reply"] == intact_targets[row["case_id"]]
        ),
        "irrelevant_original_exact_rate": rate(
            "irrelevant", lambda row: row["observed_reply"] == intact_targets[row["case_id"]]
        ),
        "swapped_original_exact_rate": rate(
            "swapped", lambda row: row["observed_reply"] == intact_targets[row["case_id"]]
        ),
        "intact_termination_rate": rate("intact", lambda row: row["terminated"]),
        "swapped_termination_rate": rate("swapped", lambda row: row["terminated"]),
        "intact_history_completion_rate": rate("intact", lambda row: row["history_completed"]),
        "reset_history_completion_rate": rate("reset", lambda row: row["history_completed"]),
        "swapped_history_completion_rate": rate("swapped", lambda row: row["history_completed"]),
        "irrelevant_history_completion_rate": rate("irrelevant", lambda row: row["history_completed"]),
        "input_target_absence_rate": float(input_absence),
        "constant_intact_exact_floor": constant_floor,
        "rows": rows,
    }
    report["probe_id"] = canonical_sha256(report)
    return report


def decide_delayed_soul_recall_probe(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless delayed recall is complete and content-specific."""

    failures: list[dict[str, Any]] = []
    raw_case_count = report.get("case_count")
    if isinstance(raw_case_count, bool) or not isinstance(raw_case_count, int) or raw_case_count < _MINIMUM_HELDOUT_CASES:
        failures.append(
            {
                "metric": "case_count",
                "observed": raw_case_count,
                "required_minimum": _MINIMUM_HELDOUT_CASES,
            }
        )

    required_exact = {
        "intact_exact_rate": 1.0,
        "swapped_exact_rate": 1.0,
        "intact_termination_rate": 1.0,
        "swapped_termination_rate": 1.0,
        "intact_history_completion_rate": 1.0,
        "reset_history_completion_rate": 1.0,
        "swapped_history_completion_rate": 1.0,
        "irrelevant_history_completion_rate": 1.0,
        "input_target_absence_rate": 1.0,
    }
    required_zero = {
        "reset_original_exact_rate",
        "irrelevant_original_exact_rate",
        "swapped_original_exact_rate",
    }

    def probability(name: str) -> float | None:
        raw = report.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    for name, required in required_exact.items():
        observed = probability(name)
        if observed != required:
            failures.append({"metric": name, "observed": observed, "required": required})
    for name in required_zero:
        observed = probability(name)
        if observed != 0.0:
            failures.append({"metric": name, "observed": observed, "required": 0.0})

    intact = probability("intact_exact_rate")
    floor = probability("constant_intact_exact_floor")
    controls = {
        name: probability(name)
        for name in required_zero
    }
    if intact is None or floor is None or intact <= floor:
        failures.append(
            {
                "metric": "intact_exact_vs_constant_floor",
                "observed": intact,
                "required_strictly_greater_than": floor,
            }
        )
    for name in sorted(required_zero):
        control = controls[name]
        if intact is None or control is None or intact <= control:
            failures.append(
                {
                    "metric": "intact_exact_vs_" + name,
                    "observed": intact,
                    "required_strictly_greater_than": control,
                }
            )
    decision = {
        "schema": SOUL_DELAYED_RECALL_GATE_SCHEMA,
        "passed": not failures,
        "required_minimum_cases": _MINIMUM_HELDOUT_CASES,
        "failures": failures,
    }
    decision["gate_id"] = canonical_sha256(decision)
    return decision


__all__ = [
    "SOUL_DELAYED_RECALL_CASE_SCHEMA",
    "SOUL_DELAYED_RECALL_GATE_SCHEMA",
    "SOUL_DELAYED_RECALL_PROBE_SCHEMA",
    "SoulDelayedRecallCase",
    "build_delayed_soul_recall_cases",
    "decide_delayed_soul_recall_probe",
    "run_delayed_soul_recall_probe",
]
