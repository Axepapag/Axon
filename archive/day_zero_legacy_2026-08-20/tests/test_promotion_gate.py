import math

import pytest

from training.promotion_gate import (
    evaluate_pilot_gate,
    evaluate_promotion_gate,
    exact_v3_pilot_gate,
)


def test_promotion_gate_passes_matching_metrics() -> None:
    gate = {
        "metrics": {
            "copy": {"exact_fill": {"min": 1.0}},
            "blank": {
                "char_acc": {"min": 0.347},
                "pred_top_frac": {"max": 0.85},
            },
        }
    }
    metrics = {
        "copy": {"exact_fill": 1.0},
        "blank": {"char_acc": 0.35, "pred_top_frac": 0.80},
    }

    assert evaluate_promotion_gate(metrics, gate) == {"passed": True, "violations": []}


def test_promotion_gate_fails_closed_with_precise_violations() -> None:
    gate = {
        "metrics": {
            "blank": {
                "char_acc": {"min": 0.347},
                "pred_top_frac": {"max": 0.85},
                "pred_unique": {"min": 20},
            },
            "partial": {"suffix_exact": {"min": 0.277}},
        }
    }
    metrics = {
        "blank": {"char_acc": 0.30, "pred_top_frac": 0.90},
        "partial": {},
    }

    result = evaluate_promotion_gate(metrics, gate)

    assert result["passed"] is False
    assert result["violations"] == [
        "blank.char_acc=0.3 below minimum 0.347",
        "blank.pred_top_frac=0.9 above maximum 0.85",
        "missing metric: blank.pred_unique",
        "missing metric: partial.suffix_exact",
    ]


def test_promotion_gate_rejects_non_finite_values() -> None:
    gate = {"metrics": {"blank": {"char_acc": {"min": math.inf}}}}
    metrics = {"blank": {"char_acc": math.nan}}

    result = evaluate_promotion_gate(metrics, gate)

    assert result == {
        "passed": False,
        "violations": ["non-finite metric: blank.char_acc"],
    }


SUITE_SHA256 = "a" * 64


def _fixed_eval(step: int, *, suite_sha256: str = SUITE_SHA256) -> dict:
    return {
        "schema": "axon_charslot_fixed_eval_v1",
        "step": step,
        "fixed_suite_sha256": suite_sha256,
        "metrics": {
            "copy": {"exact_fill": 0.95, "char_acc": 0.99},
            "partial": {"suffix_exact": 0.20},
            "blank": {
                "char_acc": 0.27,
                "suffix_char_acc": 0.20,
                "pred_top_frac": 0.75,
                "pred_unique": 30,
                "collapse_var": 1e-6,
            },
        },
    }


def test_exact_v3_pilot_is_continuable_but_not_implicitly_promotable() -> None:
    gate = exact_v3_pilot_gate(suite_sha256=SUITE_SHA256)

    result = evaluate_pilot_gate(_fixed_eval(250_000), _fixed_eval(252_000), gate)

    assert result["passed"] is True
    assert result["continuable"] is True
    assert result["promotable"] is False
    assert result["violations"] == []
    assert result["promotion_violations"] == ["pilot gate does not authorize promotion"]


def test_exact_v3_pilot_fails_closed_on_source_suite_or_final_nonfinite() -> None:
    gate = exact_v3_pilot_gate(suite_sha256=SUITE_SHA256)
    source = _fixed_eval(250_000, suite_sha256="b" * 64)
    final = _fixed_eval(252_000)
    final["metrics"]["blank"]["collapse_var"] = math.nan

    result = evaluate_pilot_gate(source, final, gate)

    assert result["continuable"] is False
    assert result["promotable"] is False
    assert "source fixed-suite SHA-256 does not match the gate" in result["violations"]
    assert "final non-finite metric: blank.collapse_var" in result["violations"]


def test_exact_v3_pilot_factory_rejects_ambiguous_contracts() -> None:
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        exact_v3_pilot_gate(suite_sha256="not-a-hash")
    with pytest.raises(ValueError, match="requires promotion_metrics"):
        exact_v3_pilot_gate(suite_sha256=SUITE_SHA256, promotion_authorized=True)
