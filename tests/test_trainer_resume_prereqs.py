from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import pytest
import torch

from training.trainer_slot import (
    Phase0bCurriculum,
    build_fixed_eval_envelope,
    capture_training_state,
    choose_draft_mode,
    load_optimizer_state_with_lr_policy,
    restore_training_state,
)


FAMILIES = (
    "runtime_response_delta_chain_v3",
    "structured_knowledge_delta_chain_v3",
    "scratchpad_delta_chain_v3",
)


def _row(family: str, index: int, split: str, *, allowed: list[str] | None = None) -> dict:
    row = {
        "schema": "axon_phase0b_delta_v3",
        "family": family,
        "split": split,
        "source": {"record_id": f"{family}-{split}-{index}"},
        "active_field": {
            "conversation_history": f"history {family} {index}",
            "user_input": f"question {index}",
            "response_draft": "",
        },
        "target_delta": {
            "region": "response_draft",
            "op": "replace",
            "text": f"answer {family[-8:]} {index}",
        },
        "budget_audit": {"silent_clips": 0, "unsupported_substitutions": 0},
    }
    if allowed is not None:
        row["allowed_draft_modes"] = allowed
    return row


def _write_exact_v3(tmp_path, *, train_count: int = 1) -> None:
    for family in FAMILIES:
        rows = [_row(family, index, "train", allowed=["blank"]) for index in range(train_count)]
        rows.extend(_row(family, index, "dev") for index in range(1, 4))
        (tmp_path / f"{family}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _curriculum(tmp_path, seed: int, *, weighted: bool = False) -> Phase0bCurriculum:
    return Phase0bCurriculum(
        str(tmp_path),
        ",".join(FAMILIES),
        max_chars=64,
        rng=random.Random(seed),
        family_sampling="weighted" if weighted else "global",
        family_weights="0.30,0.40,0.30" if weighted else None,
    )


def test_exact_v3_eval_is_stable_and_stratified_for_64_cases(tmp_path) -> None:
    _write_exact_v3(tmp_path)
    first = _curriculum(tmp_path, 17).eval_examples(64)
    second = _curriculum(tmp_path, 999).eval_examples(64)

    assert [row["eval_case_id"] for row in first] == [row["eval_case_id"] for row in second]
    assert Counter(row["family"] for row in first) == {
        FAMILIES[0]: 22,
        FAMILIES[1]: 21,
        FAMILIES[2]: 21,
    }


def test_exact_v3_allowed_draft_modes_constrain_sampling(tmp_path) -> None:
    _write_exact_v3(tmp_path)
    curriculum = _curriculum(tmp_path, 17)
    blank_only = next(row for row in curriculum.examples if row["allowed_draft_modes"] == ["blank"])

    assert choose_draft_mode(blank_only, [0.2, 0.3, 0.5], rng=random.Random(2)) == "blank"
    with pytest.raises(ValueError, match="zero probability"):
        choose_draft_mode(blank_only, [1.0, 1.0, 0.0], rng=random.Random(2))


def test_weighted_family_sampler_is_exact_nonrepeating_and_resumable(tmp_path) -> None:
    _write_exact_v3(tmp_path, train_count=12)
    first = _curriculum(tmp_path, 17, weighted=True)
    prefix = [first.next() for _ in range(10)]

    assert Counter(row["family"] for row in prefix) == {
        FAMILIES[0]: 3,
        FAMILIES[1]: 4,
        FAMILIES[2]: 3,
    }
    assert len({row["source"] for row in prefix}) == len(prefix)

    for _ in range(3):
        first.next()
    state = first.state_dict()
    expected = [first.next()["source"] for _ in range(8)]
    resumed = _curriculum(tmp_path, 17, weighted=True)
    resumed.load_state_dict(state)

    assert [resumed.next()["source"] for _ in range(8)] == expected
    assert state["family_sampling_policy"] == "weighted"
    assert state["family_weights"] == dict(zip(FAMILIES, (0.3, 0.4, 0.3)))
    assert state["family_exhaustion_policy"] == "error"


def test_weighted_family_sampler_fails_instead_of_cycling(tmp_path) -> None:
    _write_exact_v3(tmp_path, train_count=1)
    curriculum = _curriculum(tmp_path, 17, weighted=True)
    for _ in range(3):
        curriculum.next()

    with pytest.raises(RuntimeError, match="exhausted without replacement"):
        curriculum.next()


def test_optimizer_override_preserves_moments_and_reapplies_requested_lr() -> None:
    source_param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    source_opt = torch.optim.AdamW([source_param], lr=1e-3)
    source_param.grad = torch.tensor([0.25, -0.5])
    source_opt.step()
    state = source_opt.state_dict()
    source_moment = next(iter(state["state"].values()))["exp_avg"].clone()

    resumed_param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    resumed_opt = torch.optim.AdamW([resumed_param], lr=9e-4)
    audit = load_optimizer_state_with_lr_policy(
        resumed_opt,
        state,
        requested_lr=1e-4,
        policy="override",
    )

    resumed_state = next(iter(resumed_opt.state.values()))
    assert torch.equal(resumed_state["exp_avg"], source_moment)
    assert resumed_opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert audit == {
        "schema": "axon_optimizer_resume_v1",
        "state_restored": True,
        "lr_policy": "override",
        "requested_lr": 1e-4,
        "source_lrs": [1e-3],
        "loaded_lrs": [1e-3],
        "effective_lrs": [1e-4],
    }


class _Sampler:
    def __init__(self, family: str, count: int) -> None:
        self.family = family
        self.count = count
        self.loaded = False

    def state_dict(self) -> dict:
        return {
            "schema": "axon_phase0b_sampler_v1",
            "position": 0,
            "order": list(range(self.count)),
            "rng_state": random.Random(3).getstate(),
            "example_count": self.count,
            "families": [self.family],
        }

    def load_state_dict(self, state: dict) -> None:
        self.loaded = True


def test_training_state_can_auditably_reset_only_the_curriculum_sampler() -> None:
    source = _Sampler("exact_v2", 8)
    active = _Sampler("exact_v3", 5)
    outer_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        saved = capture_training_state(source)
        audit = restore_training_state(saved, active, curriculum_state_policy="reset")
    finally:
        random.setstate(outer_state[0])
        np.random.set_state(outer_state[1])
        torch.set_rng_state(outer_state[2])

    assert active.loaded is False
    assert audit["intentionally_reset"] is True
    assert audit["restored"] is False
    assert audit["source"]["families"] == ["exact_v2"]
    assert audit["active"]["families"] == ["exact_v3"]


def test_source_and_final_fixed_eval_envelopes_share_the_exact_suite() -> None:
    suite_sha256 = "a" * 64
    metrics = {
        mode: {"case_set_sha256": suite_sha256, "char_acc": 0.5}
        for mode in ("copy", "partial", "blank")
    }

    source = build_fixed_eval_envelope(
        step=250000,
        fixed_suite_sha256=suite_sha256,
        metrics=metrics,
        metadata={"role": "pre_update_source"},
    )
    final = build_fixed_eval_envelope(
        step=252000,
        fixed_suite_sha256=suite_sha256,
        metrics=metrics,
        metadata={"role": "post_update_final"},
    )

    assert source["schema"] == final["schema"] == "axon_charslot_fixed_eval_v1"
    assert source["fixed_suite_sha256"] == final["fixed_suite_sha256"] == suite_sha256
    assert source["step"] == 250000
    assert final["step"] == 252000
