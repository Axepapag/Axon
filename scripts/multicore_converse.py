#!/usr/bin/env python3
"""Multi-core shared-field conversational demo for Axon exact-v4 64D.

One shared AxonCore (loaded once from a conversational checkpoint) drives N
logical cores. Each core carries its OWN private soul (cloned from the
checkpoint soul at boot) and inhales/exhales it every tick.

Per user turn the cores tick recursively over ONE shared field:
  tick t: every core attends the field (history + committed council deltas
          + user input) and proposes a response delta, exhaling an updated
          soul that it carries to the next tick.
          A consolidation gate (confidence = mean max-prob over non-empty
          positions) picks the winning delta, which commits to the shared
          field's history region as a "Council:" line.
  tick t+1: every core attends the updated field — i.e. every core attends
          over the other cores' committed deltas.
Halt when the winning delta repeats for `stable_ticks` consecutive ticks or
the tick budget is spent. The final committed delta is Axon's utterance.

Diversity: the checkpoint's weights are shared, so each core's soul is
perturbed once at boot with small seeded Gaussian noise (`--soul-noise`).
That makes each core a genuine individual — same body, different state.

The response region is left empty at inference (the checkpoint was trained
with blank drafts), and council deltas ride in the history region, which IS
the trained input distribution. Draft-in-response-region refinement needs a
future curriculum that trains non-empty drafts.

Layout matches training exactly (history 0-127 / user 128-191 / response
192-255) via CharSlotFieldBuilder — no checkpoint surgery required.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cores.core import AxonCore, CoreConfig
from training.conversational_objective import decode_parallel_logits
from runtime.field.charslot import CharSlotFieldBuilder

DEFAULT_CKPT = ROOT / "runs" / "conversational_cpu_autopilot" / "ckpt_440500.pt"
HISTORY_CHARS = 128
USER_CHARS = 64
RESP_CHARS = 64


def load_core(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if "core_state" not in payload or "cfg" not in payload:
        raise ValueError(f"checkpoint {path} missing core_state or cfg")
    cfg = CoreConfig.from_dict(payload["cfg"])
    core = AxonCore(cfg).to(device)
    core.load_state_dict(payload["core_state"])
    core.eval()
    soul = payload["soul_state"]["soul"].to(device)
    soul_mask = payload["soul_state"]["soul_mask"].to(device)
    return core, soul, soul_mask, payload.get("step", "?")


def propose(core, builder, soul, soul_mask, history, user_input, temperature: float = 0.0,
            generator: torch.Generator | None = None):
    """One core attends the shared field and proposes a delta.

    temperature=0 -> deterministic argmax decode (the trained path).
    temperature>0 -> per-position sampling at that temperature; confidence is
    the mean probability of the SAMPLED tokens over non-empty positions, so
    cores that sample low-probability tokens are penalized by the gate.

    Returns (text, confidence, new_soul).
    """
    built = builder.build(history, user_input, "", "")
    with torch.no_grad():
        out = core.forward_charslot(
            built["field16"],
            built["region"],
            soul,
            soul_mask=soul_mask,
            response_slice=builder.resp_slice,
        )
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        new_soul = out["soul"].detach() if out.get("soul") is not None else soul
        if temperature <= 0.0:
            decoded = decode_parallel_logits(logits)
            text = decoded.texts[0]
            probs = torch.softmax(logits, dim=-1)
            top = probs.max(dim=-1)
            nonempty = top.indices[0] != builder.empty_index
            conf = float(top.values[0][nonempty].mean().item()) if nonempty.any() else 0.0
            return text, conf, new_soul
        probs = torch.softmax(logits[0] / temperature, dim=-1)  # (resp_chars, n_classes)
        if generator is not None:
            sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            sampled = torch.multinomial(probs, 1).squeeze(-1)
        # first-empty termination, matching decode_parallel_logits semantics
        chars = []
        token_probs = []
        for pos, idx in enumerate(sampled.tolist()):
            if idx == builder.empty_index:
                break
            chars.append(builder.bank.chars[idx])
            token_probs.append(float(probs[pos, idx].item()))
        text = "".join(chars)
        conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.0
    return text, conf, new_soul


def converse_turn(
    core,
    builder,
    souls: list[torch.Tensor],
    soul_masks: list[torch.Tensor],
    history: str,
    user_input: str,
    max_ticks: int,
    stable_ticks: int,
    temperatures: list[float] | None = None,
    generators: list[torch.Generator] | None = None,
    verbose: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the recursive multi-core tick loop for one user turn.

    Committed deltas ride in the history region ("Council:" lines) so every
    core attends over the other cores' deltas on the next tick.
    """
    deltas: list[str] = []
    stable = 0
    trace: list[dict[str, Any]] = []
    n = len(souls)
    draft = ""
    for tick in range(max_ticks):
        council = "".join(f"\nCouncil: {d}" for d in deltas)
        field_history = (history + council)[-HISTORY_CHARS * 2 :]
        proposals = []
        for i in range(n):
            temp = temperatures[i] if temperatures else 0.0
            gen = generators[i] if generators else None
            text, conf, new_soul = propose(
                core, builder, souls[i], soul_masks[i], field_history, user_input,
                temperature=temp, generator=gen,
            )
            souls[i] = new_soul  # exhale -> carry -> inhale next tick
            proposals.append({"core": i, "text": text, "conf": conf})
        # consolidation gate: highest confidence wins the commit
        best = max(range(n), key=lambda i: proposals[i]["conf"])
        winner = proposals[best]["text"]
        if winner == draft:
            stable += 1
        else:
            stable = 0
            draft = winner
            deltas.append(winner)
        trace.append({"tick": tick, "winner": best, "draft": draft, "proposals": proposals})
        if verbose:
            print(f"  tick {tick}: committed={draft!r} (core {best}, conf {proposals[best]['conf']:.3f})")
            for p in proposals:
                mark = "*" if p["core"] == best else " "
                print(f"    {mark} core {p['core']:2d} [{p['conf']:.3f}] {p['text']!r}")
        if stable >= stable_ticks:
            break
    return draft, trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Axon multi-core shared-field conversation demo")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cores", type=int, default=10)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--stable-ticks", type=int, default=2)
    parser.add_argument("--prompt", type=str, default=None, help="one-shot prompt instead of interactive loop")
    parser.add_argument("--soul-noise", type=float, default=0.05,
                        help="Gaussian noise scale (fraction of soul std) differentiating each core's soul at boot")
    parser.add_argument("--temperature-spread", type=float, default=0.0,
                        help="if >0, core i samples at temperatures spread "
                             "linspace(1-spread, 1+spread) instead of argmax")
    parser.add_argument("--quiet", action="store_true", help="hide per-tick proposal table")
    parser.add_argument("--log", type=Path, default=None, help="append JSONL turn traces here")
    args = parser.parse_args()

    device = torch.device(args.device)
    core, soul, soul_mask, step = load_core(args.checkpoint, device)
    builder = CharSlotFieldBuilder(HISTORY_CHARS, USER_CHARS, RESP_CHARS, device, torch.float32)

    # N logical cores: shared weights, private souls cloned from the checkpoint
    # and individually perturbed — same body, different state. Diversity comes
    # from initialization, never from constraints (Triad v2 lesson).
    souls = []
    for i in range(args.cores):
        g = torch.Generator(device="cpu").manual_seed(1000 + i)
        noise = torch.randn(soul.shape, generator=g).to(device, soul.dtype)
        souls.append(soul + noise * float(soul.std().item()) * args.soul_noise)
    soul_masks = [soul_mask.clone() for _ in range(args.cores)]

    # Optional sampling diversity: per-core temperature + private RNG stream.
    temperatures = None
    generators = None
    if args.temperature_spread > 0.0 and args.cores > 1:
        lo, hi = 1.0 - args.temperature_spread, 1.0 + args.temperature_spread
        temperatures = [lo + (hi - lo) * i / (args.cores - 1) for i in range(args.cores)]
        generators = [torch.Generator(device=device).manual_seed(2000 + i) for i in range(args.cores)]
        print("core temperatures:", [f"{t:.2f}" for t in temperatures])

    print(f"Axon council online: {args.cores} cores, shared 64D weights (step {step}), device {device}")
    print(f"Tick budget {args.ticks}, halt after {args.stable_ticks} stable ticks. Type 'quit' to exit.")
    print("-" * 60)

    history = ""
    turns = 0
    while True:
        if args.prompt is not None:
            user_input = args.prompt
        else:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not user_input:
            if args.prompt is not None:
                break
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        t0 = time.time()
        draft, trace = converse_turn(
            core, builder, souls, soul_masks, history, user_input,
            args.ticks, args.stable_ticks,
            temperatures=temperatures, generators=generators,
            verbose=not args.quiet,
        )
        dt = time.time() - t0
        print(f"Axon: {draft}")
        print(f"  [{len(trace)} ticks, {dt:.1f}s]")
        history = (history + f"\nUser: {user_input}\nAssistant: {draft}")[-HISTORY_CHARS * 2 :]
        turns += 1
        if args.log is not None:
            with args.log.open("a", encoding="utf-8") as h:
                h.write(json.dumps({"turn": turns, "user": user_input, "response": draft,
                                    "ticks": len(trace), "trace": trace}, ensure_ascii=False) + "\n")
        if args.prompt is not None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
