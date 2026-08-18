#!/usr/bin/env python3
"""Bounded conversational runtime prototype for Axon.

Loads the finished 64D conversational CPU checkpoint and decodes responses
for seed and held-out prompts. This script does not train, mutate state, or
start a daemon. It proves whether the checkpoint generalizes beyond the
memorized seed set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cores.core import AxonCore, CoreConfig
from training.conversational_objective import decode_parallel_logits
from training.trainer_slot import CharSlotFieldBuilder


DEFAULT_CKPT = ROOT / "runs" / "conversational_cpu_autopilot" / "ckpt_440500.pt"
HISTORY_CHARS = 128
USER_CHARS = 64
RESP_CHARS = 64


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "core_state" not in payload or "cfg" not in payload:
        raise ValueError(f"checkpoint {path} missing core_state or cfg")
    return payload


def build_core(payload: dict[str, Any], device: torch.device) -> AxonCore:
    cfg = CoreConfig.from_dict(payload["cfg"])
    core = AxonCore(cfg).to(device)
    core.load_state_dict(payload["core_state"])
    core.eval()
    return core


def load_soul_state(payload: dict[str, Any], device: torch.device):
    soul_state = payload.get("soul_state")
    if not isinstance(soul_state, dict):
        return None, None
    soul = soul_state.get("soul")
    soul_mask = soul_state.get("soul_mask")
    if soul is None or soul_mask is None:
        return None, None
    return soul.to(device), soul_mask.to(device)


def decode_response(
    core: AxonCore,
    builder: CharSlotFieldBuilder,
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
    history: str,
    user_input: str,
) -> str:
    built = builder.build(history, user_input, "", "")
    field16 = built["field16"]
    region = built["region"]
    with torch.no_grad():
        out = core.forward_charslot(
            field16,
            region,
            soul,
            soul_mask=soul_mask,
            response_slice=builder.resp_slice,
        )
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        decoded = decode_parallel_logits(logits)
    return decoded.texts[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversational runtime prototype")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    payload = load_checkpoint(args.checkpoint, device)
    core = build_core(payload, device)
    soul, soul_mask = load_soul_state(payload, device)

    builder = CharSlotFieldBuilder(
        history_chars=HISTORY_CHARS,
        user_chars=USER_CHARS,
        resp_chars=RESP_CHARS,
        device=device,
        dtype=torch.float32,
    )

    seed_prompts = [
        ("What time is it?", "I do not know the time."),
        ("What is your favorite color?", "I like blue best."),
        ("How are you?", "I am doing well."),
        ("Thank you", "You are welcome."),
        ("How old are you?", "I am a young AI."),
        ("Hello.", "Hello there."),
        ("Can you help me?", "Yes I can help."),
        ("Is it raining?", "I cannot see outside."),
        ("What is your name?", "My name is Axon."),
        ("Goodbye.", "Goodbye for now."),
    ]

    held_out_prompts = [
        "What is the weather today?",
        "Tell me a joke.",
        "What is two plus two?",
        "Why is the sky blue?",
        "What do you think about robots?",
        "Can you write a poem?",
        "What is your purpose?",
        "Do you like music?",
        "What is the capital of France?",
        "How do you learn new things?",
    ]

    results: list[dict[str, Any]] = []
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Step: {payload.get('step', '?')}")
    print(f"Schema: {payload.get('checkpoint_schema', '?')}")
    print(f"Core config: d_model={core.cfg.d_model}, layers={core.cfg.n_layers}, heads={core.cfg.n_heads}")
    print(f"Soul present: {soul is not None}")
    print("-" * 60)

    print("SEED PROMPTS (memorization check):")
    seed_correct = 0
    for prompt, expected in seed_prompts:
        response = decode_response(core, builder, soul, soul_mask, "", prompt)
        match = response.strip() == expected.strip()
        seed_correct += int(match)
        marker = "OK" if match else "MISMATCH"
        print(f"  [{marker}] {prompt!r} -> {response!r}")
        results.append({
            "kind": "seed",
            "prompt": prompt,
            "expected": expected,
            "response": response,
            "exact_match": match,
        })
    print(f"Seed exact-match: {seed_correct}/{len(seed_prompts)}")
    print("-" * 60)

    print("HELD-OUT PROMPTS (generalization check):")
    nonempty = 0
    for prompt in held_out_prompts:
        response = decode_response(core, builder, soul, soul_mask, "", prompt)
        if response.strip():
            nonempty += 1
        print(f"  {prompt!r} -> {response!r}")
        results.append({
            "kind": "held_out",
            "prompt": prompt,
            "response": response,
            "nonempty": bool(response.strip()),
        })
    print(f"Held-out nonempty responses: {nonempty}/{len(held_out_prompts)}")
    print("-" * 60)

    print("MULTI-TURN PROMPTS (history sensitivity check):")
    multi_turn_prompts = [
        ("Hello.", "Hello there."),
        ("Hello.\nAssistant: Hello there.\nUser: How are you?", "I am doing well."),
        ("What is your name?\nAssistant: My name is Axon.\nUser: What time is it?", "I do not know the time."),
    ]
    multi_correct = 0
    for history, expected in multi_turn_prompts:
        # history already contains the prior turn(s); user_input is a short follow-up
        lines = [line.strip() for line in history.split("\n") if line.strip()]
        user_input = lines[-1].replace("User: ", "") if lines else ""
        prior = "\n".join(lines[:-1])
        response = decode_response(core, builder, soul, soul_mask, prior, user_input)
        match = response.strip() == expected.strip()
        multi_correct += int(match)
        marker = "OK" if match else "MISMATCH"
        print(f"  [{marker}] history={prior!r} + {user_input!r} -> {response!r}")
        results.append({
            "kind": "multi_turn",
            "history": prior,
            "user_input": user_input,
            "expected": expected,
            "response": response,
            "exact_match": match,
        })
    print(f"Multi-turn exact-match: {multi_correct}/{len(multi_turn_prompts)}")

    if args.json:
        out_path = Path("conversational_runtime_prototype_results.json")
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "step": payload.get("step"),
                    "schema": payload.get("checkpoint_schema"),
                    "seed_correct": seed_correct,
                    "seed_total": len(seed_prompts),
                    "held_out_nonempty": nonempty,
                    "held_out_total": len(held_out_prompts),
                    "multi_turn_correct": multi_correct,
                    "multi_turn_total": len(multi_turn_prompts),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
