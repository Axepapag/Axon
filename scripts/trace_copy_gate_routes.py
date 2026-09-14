#!/usr/bin/env python3
"""Trace copy-gate logit and gradient dynamics on a tiny exact assignment.

Reference-core probe for the paused D64 architecture screen: every tournament
candidate reached teacher-forced payload-content accuracy 1.0 while copy-gate
accuracy and free-running payload transport stayed 0.0. The screen ran the
default receipt profile ``continuation_v1`` (copy-gate weight 0.25) whose
documented failure mode is "drove the learned copy/generate route to zero".
This probe reproduces the exact tournament training composition (same manifest,
stage, weights, learning rate, seed) on one small core and traces, per step:

- alignment copy-gate / position accuracy and copy-gate loss (teacher-forced)
- copy_gate bias value plus weight/bias gradient norms after backward
- mean gate logit over the supervised copy region and at EOS (forward hook)
- periodic free-running evaluation: exact typed emission + payload transport

Usage:
    python scripts/trace_copy_gate_routes.py --profile continuation_v1
    python scripts/trace_copy_gate_routes.py --profile route_eos_balanced_v2

Read-only with respect to canonical State; no serving, promotion, or ledger
mutation. A JSON summary is printed to stdout for receipting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.soul import SoulSnapshot, empty_soul_layers
from training.first_form_curriculum import load_first_form_curriculum
from training.foundation_motor_curriculum import (
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    apply_receipt_continuation_teach_weights,
    foundation_motor_v2_action,
    foundation_motor_v2_stage_policy,
    is_foundation_motor_v2_episode,
    oversample_multicell_copy_cases,
    receipt_continuation_teach_profile,
)
from training.living_reasoning_curriculum import (
    evaluate_living_episode,
    living_episode_objective,
)
from training.living_reasoning_d64 import (
    LivingReasoningCoreD64,
    candidate_a_config,
)

DEFAULT_MANIFEST = (
    ROOT
    / "State"
    / "training"
    / "curricula"
    / "ffcs_v1"
    / "a872278fd0e8ef926370e1712d01dcf0a277672c4a0088af44aef483d8417740"
    / "manifest.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
            RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
            RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
        ),
        default=RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    )
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--generate-gate-bias",
        type=float,
        default=1.5,
        help="initial copy-gate bias; initialization only, never architecture identity",
    )
    parser.add_argument(
        "--copy-gate-weight",
        type=float,
        default=None,
        help="override the alignment_copy_gate component weight (experiment knob)",
    )
    parser.add_argument(
        "--eos-gate-weight",
        type=float,
        default=None,
        help="override the alignment_eos_gate component weight (experiment knob)",
    )
    parser.add_argument(
        "--eos-generate-head-route",
        action="store_true",
        help="opt the receipt architecture into generate-head EOS routing",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument(
        "--train-case-limit",
        type=int,
        default=None,
        help="optional deterministic prefix for a bounded motor-overfit probe",
    )
    parser.add_argument(
        "--heldout-case-limit",
        type=int,
        default=None,
        help="optional deterministic prefix for faster disjoint evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval-every must be positive")
    if args.train_case_limit is not None and args.train_case_limit < 1:
        raise ValueError("train-case-limit must be positive")
    if args.heldout_case_limit is not None and args.heldout_case_limit < 1:
        raise ValueError("heldout-case-limit must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    curricula = load_first_form_curriculum(args.manifest)
    stage = foundation_motor_v2_stage_policy("copy_alignment")
    eligible = set(stage["eligible_actions"])

    def _motor_cases(split: str) -> tuple[object, ...]:
        cases = tuple(
            case
            for item in (curricula,)
            for case in item.teaching_cases
            if case.family == "F0"
            and case.episode.split == split
            and is_foundation_motor_v2_episode(case.episode)
            and foundation_motor_v2_action(case.episode) in eligible
        )
        if split == "train":
            cases = oversample_multicell_copy_cases(cases)
        return cases

    train_cases = _motor_cases("train")
    heldout_cases = _motor_cases("heldout")
    if args.train_case_limit is not None:
        train_cases = train_cases[: args.train_case_limit]
    if args.heldout_case_limit is not None:
        heldout_cases = heldout_cases[: args.heldout_case_limit]
    if not train_cases:
        raise RuntimeError("no copy_alignment train cases resolved from the manifest")
    if not heldout_cases:
        raise RuntimeError("no copy_alignment heldout cases resolved from the manifest")

    # Exact tournament training composition (mirrors train_living_reasoning_smoke).
    component_weights = dict(stage["component_weights"])
    component_weights = apply_receipt_continuation_teach_weights(
        component_weights,
        training_stage="copy_alignment",
        receipt_teaching_profile=args.profile,
    )
    if args.copy_gate_weight is not None:
        component_weights["alignment_copy_gate"] = float(args.copy_gate_weight)
    if args.eos_gate_weight is not None:
        component_weights["alignment_eos_gate"] = float(args.eos_gate_weight)
    position_reduction = str(
        receipt_continuation_teach_profile(args.profile)["alignment_position_reduction"]
    )

    config = candidate_a_config(
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=0.0,
        receipt_continuation=True,
        generate_gate_bias=args.generate_gate_bias,
        eos_generate_head_route=args.eos_generate_head_route,
    )
    model = LivingReasoningCoreD64(config).to(device)
    soul = SoulSnapshot(
        core_id="core-a",
        architecture_id=model.architecture_id,
        parameter_generation="g0",
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )

    gate_logit_trace: list[torch.Tensor] = []

    def _capture_gate_logits(module, _inputs, output) -> None:
        gate_logit_trace.append(output.detach().reshape(-1))

    hook = model.copy_gate.register_forward_hook(_capture_gate_logits)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    traces: list[dict[str, float]] = []
    eval_rows: list[dict[str, float]] = []

    decode_diags: list[dict[str, Any]] = []

    def _run_eval(step: int, *, final: bool = False) -> None:
        model.eval()
        keys = (
            "alignment_copy_gate_accuracy",
            "alignment_position_accuracy",
            "alignment_eos_gate_accuracy",
            "payload_transport_exact_rate",
            "typed_emission_exact_rate",
            "payload_teacher_forced_token_accuracy",
            "payload_teacher_forced_content_accuracy",
            "payload_teacher_forced_eos_accuracy",
            "heldout_mean_loss",
        )
        sums = {key: 0.0 for key in keys}
        for case in heldout_cases:
            result = evaluate_living_episode(
                model,
                case.episode,
                soul,
                core_id="core-a",
                parameter_generation="g0",
            )
            for key in keys:
                if key in result:
                    sums[key] += float(result[key])
            if final and args.eos_generate_head_route:
                decode_diags.append(_decode_diag(model, case, soul))
        row = {"step": float(step)}
        for key in keys:
            if key in result:
                row[key] = sums[key] / len(heldout_cases)
        eval_rows.append(row)
        model.train()

    def _decode_diag(model: LivingReasoningCoreD64, case: Any, soul: SoulSnapshot) -> dict[str, Any]:
        """Free-running receipt decode failure analysis for one heldout case."""

        from runtime.field import D64FieldCompiler
        from training.living_reasoning_curriculum import ReasoningDecision

        compiled = D64FieldCompiler().compile(case.episode.snapshot)
        unroll = model.unroll_runtime_phases(
            initial_soul=soul,
            expected_core_id="core-a",
            parameter_generation="g0",
            tick_uid=f"diag:{case.episode.episode_id}",
            canonical=compiled,
            first_workspace_text=case.episode.first_workspace_text,
            refined_workspace_text=case.episode.refined_workspace_text,
        )
        routes: dict[str, int] = {}
        diag: dict[str, Any] = {"case": case.episode.episode_id}
        for output, target in zip(unroll.outputs, case.episode.targets, strict=True):
            if target.supervision_weight <= 0 or target.decision is not ReasoningDecision.DELTA:
                continue
            execution = model.initial_decoder_execution_state(output)
            decoded = model.advance_decoder_execution(
                output,
                execution,
                work_units=512,
            )
            for item in decoded.state.trace:
                routes[item.route.value] = routes.get(item.route.value, 0) + 1
            diag.update(
                {
                    "payload_target": target.payload,
                    "payload_decoded": decoded.text,
                    "complete": bool(decoded.complete),
                    "malformed_reason": decoded.state.malformed_reason,
                    "trace_steps": len(decoded.state.trace),
                    "routes": dict(routes),
                }
            )
            break
        return diag

    started = time.perf_counter()
    for step in range(args.steps):
        case = train_cases[step % len(train_cases)]
        gate_logit_trace.clear()
        loss, _unroll, phase_metrics = living_episode_objective(
            model,
            case.episode,
            soul,
            core_id="core-a",
            parameter_generation="g0",
            component_weights=component_weights,
            alignment_position_reduction=position_reduction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        weight_grad = float(model.copy_gate.weight.grad.norm().item()) if model.copy_gate.weight.grad is not None else 0.0
        bias_grad = float(model.copy_gate.bias.grad.abs().sum().item()) if model.copy_gate.bias.grad is not None else 0.0
        optimizer.step()

        merged: dict[str, float] = {}
        for metrics in phase_metrics:
            for key, value in metrics.items():
                if "alignment" in key or key in ("payload_loss", "loss"):
                    merged[key] = float(value)
        row = {
            "step": float(step),
            "loss": float(loss.detach().item()),
            "copy_gate_bias": float(model.copy_gate.bias.detach().item()),
            "copy_gate_weight_grad_norm": weight_grad,
            "copy_gate_bias_grad_l1": bias_grad,
        }
        # Gate-logit statistics from the final-phase decode of this step.
        if gate_logit_trace:
            logits = torch.cat(gate_logit_trace)
            row["gate_logit_mean"] = float(logits.mean().item())
            row["gate_logit_min"] = float(logits.min().item())
            row["gate_logit_eos"] = float(gate_logit_trace[-1].reshape(-1)[-1].item())
            row["gate_logit_max"] = float(logits.max().item())
        for key in (
            "alignment_copy_gate_accuracy",
            "alignment_position_accuracy",
            "alignment_eos_gate_accuracy",
            "alignment_copy_gate_loss",
            "alignment_position_loss",
            "payload_loss",
        ):
            if key in merged:
                row[key] = merged[key]
        traces.append(row)

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            _run_eval(step + 1, final=step == args.steps - 1)
            last = eval_rows[-1]
            print(
                f"step {step + 1:4d} | loss {row['loss']:.4f} | "
                f"gate_acc(train) {row.get('alignment_copy_gate_accuracy', -1):.3f} | "
                f"pos_acc(train) {row.get('alignment_position_accuracy', -1):.3f} | "
                f"bias {row['copy_gate_bias']:+.4f} | wgrad {weight_grad:.2e} | "
                f"gate_acc(heldout) {last.get('alignment_copy_gate_accuracy', -1):.3f} | "
                f"transport_exact {last.get('payload_transport_exact_rate', -1):.3f}"
            )

    hook.remove()
    wall = time.perf_counter() - started
    summary = {
        "schema": "axon-copy-gate-trace-summary-v1",
        "profile": args.profile,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "generate_gate_bias": args.generate_gate_bias,
        "eos_generate_head_route": bool(args.eos_generate_head_route),
        "candidate_label": f"trace-l{args.layers}-h{args.heads}-f{args.ffn_dim}",
        "component_weights": component_weights,
        "alignment_position_reduction": position_reduction,
        "train_case_count": len(train_cases),
        "heldout_case_count": len(heldout_cases),
        "device": args.device,
        "wall_seconds": wall,
        "final_train_trace": traces[-1],
        "final_heldout_eval": eval_rows[-1],
        "decode_diags": decode_diags,
        "traces": traces,
        "eval_rows": eval_rows,
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
