"""char_slot_probe.py — decisive A/B at the field→core threshold.

Question (Jeff, 2026-07-03): does the core need the fat 8192D slot + frozen
random 8192→d_model adapter + pooled char head, or can it attend the frozen
16D character substrate directly, position-preserved, and write text back
through a per-slot 16D delta snapped to the letter bank?

Two arms, identical everything except the threshold:

  perslot : char -> frozen 16D substrate slot -> frozen orthogonal lift
            16->d_model (+ learned type/pos embeddings) -> core attends ->
            shared per-slot head d_model->16 on EACH response position ->
            cosine vs LetterBank -> char.  Position survives end to end.

  pooled  : same lift, same core, but the response head mean-pools the
            response hidden states into ONE vector and expands it into all
            max_resp characters at once — a 16D-bank miniature of the live
            ResponseDraftDeltaHead path (core.py:621). Position dies at the
            pool.  (This arm gets MORE head parameters, not fewer.)

Same Phase 0 curriculum as the live GPU runs, same copy/partial/blank eval
modes.  Trains all three modes jointly from step 1 (no teacher schedule) so
every eval is a trained skill.  CPU-runnable in minutes.

Claim under test: perslot COPY -> ~1.0 quickly; pooled COPY plateaus far
below (live runs: ~0.4-0.5).  If true, the bottleneck is the threshold, not
the core, the data, or the substrate.

Run (needs the torch-bearing interpreter):
  cd D:/Axon
  C:/Users/Jeffg/AppData/Local/Programs/Python/Python313/python.exe
      training/char_slot_probe.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import SLOT_DIM, char_to_slot, get_letter_bank  # noqa: E402
from cores.core import AxonCore, CoreConfig  # noqa: E402
from training.trainer_slot import Phase0Curriculum, draft_seed_for_mode, log  # noqa: E402


# ---------------------------------------------------------------------------
# Char-granular field builder: user_input chars + response chars, one 16D
# substrate slot per character.  No 8192D packing, no lossy adapter.
# ---------------------------------------------------------------------------
class CharFieldBuilder:
    REGION_USER = 0
    REGION_RESP = 1

    def __init__(self, max_user: int, max_resp: int, device: torch.device):
        self.max_user = max_user
        self.max_resp = max_resp
        self.n_slots = max_user + max_resp
        self.resp_slice = slice(max_user, max_user + max_resp)
        self.device = device
        self.bank = get_letter_bank()
        # frozen 16D rows for the whole alphabet, plus index lookup
        self.char_index = {c: i for i, c in enumerate(self.bank.chars)}
        self.empty_index = self.bank.empty_index
        self.bank_unit = torch.from_numpy(self.bank.vecs_unit.copy()).float().to(device)

    def _text_block(self, text: str, width: int) -> np.ndarray:
        """(width, 16): frozen char slots, zero rows beyond len(text)."""
        block = np.zeros((width, SLOT_DIM), dtype=np.float32)
        for i, ch in enumerate(text[:width]):
            block[i] = char_to_slot(ch if ch in self.char_index else " ")
        return block

    def _char_targets(self, answer: str) -> np.ndarray:
        """(max_resp,) int64 class targets over the letter bank (+<empty>)."""
        t = np.full((self.max_resp,), self.empty_index, dtype=np.int64)
        for i, ch in enumerate(answer[: self.max_resp]):
            t[i] = self.char_index.get(ch, self.char_index[" "])
        return t

    def build(self, user_input: str, answer: str, draft_text: str):
        field16 = np.zeros((self.n_slots, SLOT_DIM), dtype=np.float32)
        field16[: self.max_user] = self._text_block(user_input, self.max_user)
        field16[self.resp_slice] = self._text_block(draft_text, self.max_resp)
        region = np.zeros((self.n_slots,), dtype=np.int64)
        region[self.resp_slice] = self.REGION_RESP
        return {
            "field16": torch.from_numpy(field16).unsqueeze(0),      # (1, n, 16)
            "region": torch.from_numpy(region).unsqueeze(0),        # (1, n)
            "targets": torch.from_numpy(self._char_targets(answer)).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# The two threshold arms
# ---------------------------------------------------------------------------
def frozen_orthogonal_lift(d_model: int, seed: int = 7) -> torch.Tensor:
    """(16, d_model) with orthonormal columns: lossless over-complete lift."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d_model, SLOT_DIM))
    Q, _ = np.linalg.qr(M)  # (d_model, 16) orthonormal columns
    return torch.from_numpy(Q.astype(np.float32)).T.contiguous()  # (16, d_model)


class ThresholdModel(nn.Module):
    """Lift -> core -> response head.  head='perslot' | 'pooled'."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int,
                 n_slots: int, max_resp: int, head: str):
        super().__init__()
        self.head_kind = head
        self.max_resp = max_resp
        self.register_buffer("lift", frozen_orthogonal_lift(d_model))  # (16, d)
        self.type_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(n_slots, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.core = AxonCore(CoreConfig(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ffn_dim=ffn_dim, soul_mode="concat", soul_rows=0))
        if head == "perslot":
            # shared tiny MLP applied at EVERY response position
            self.out = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * 4, SLOT_DIM, bias=False),
            )
        elif head == "pooled":
            # miniature of ResponseDraftDeltaHead: pool -> expand all chars
            self.out = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * 4, max_resp * SLOT_DIM, bias=False),
            )
        else:
            raise ValueError(head)
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, field16: torch.Tensor, region: torch.Tensor,
                resp_slice: slice) -> torch.Tensor:
        """Returns (B, max_resp, 16) response delta vectors."""
        x = field16 @ self.lift                                  # (B, n, d)
        n = x.shape[1]
        pos = torch.arange(n, device=x.device).unsqueeze(0)
        x = x + self.type_emb(region) + self.pos_emb(pos)
        h = self.core.forward(x)["hidden"]                        # (B, n, d)
        resp_h = h[:, resp_slice, :]                              # (B, R, d)
        if self.head_kind == "perslot":
            return self.out(resp_h)                               # per position
        pooled = resp_h.mean(dim=1)                               # position dies
        return self.out(pooled).view(-1, self.max_resp, SLOT_DIM)

    def char_logits(self, delta16: torch.Tensor, bank_unit: torch.Tensor) -> torch.Tensor:
        """Cosine vs the frozen 16D letter bank -> (B, R, n_chars)."""
        v = F.normalize(delta16, dim=-1)
        return (v @ bank_unit.T) * self.temp


# ---------------------------------------------------------------------------
# Train / eval one arm
# ---------------------------------------------------------------------------
EVAL_MODES = ("copy", "partial", "blank")


@torch.no_grad()
def evaluate(model: ThresholdModel, builder: CharFieldBuilder,
             examples: list[dict], mode: str) -> dict:
    model.eval()
    bank = builder.bank
    exact = chars_total = chars_correct = 0
    sample = None
    for ex in examples:
        ans = ex["answer"]
        built = builder.build(ex["user_input"], ans, draft_seed_for_mode(ans, mode))
        delta = model(built["field16"], built["region"], builder.resp_slice)
        logits = model.char_logits(delta, builder.bank_unit)
        idx = logits.argmax(dim=-1)[0].tolist()
        pred = "".join("" if i == bank.empty_index else bank.chars[i]
                       for i in idx[: len(ans)])
        exact += int(pred.strip() == ans.strip())
        for a, b in zip(pred, ans):
            chars_total += 1
            chars_correct += int(a == b)
        if sample is None:
            sample = {"target": ans[:64], "pred": pred[:64]}
    model.train()
    t = max(1, len(examples))
    return {"exact_fill": exact / t,
            "char_acc": chars_correct / max(1, chars_total),
            "sample": sample}


def run_arm(head: str, args, curriculum: Phase0Curriculum,
            eval_examples: list[dict], device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    builder = CharFieldBuilder(args.max_user, args.max_resp, device)
    model = ThresholdModel(args.d_model, args.n_layers, args.n_heads,
                           args.ffn_dim, builder.n_slots, args.max_resp, head)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.out.parameters())
    log("BUILD", f"arm={head} d={args.d_model} l={args.n_layers} h={args.n_heads} "
                 f"ffn={args.ffn_dim} slots={builder.n_slots} "
                 f"params={n_params:,} (head={head_params:,})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mode_weights = [0.4, 0.3, 0.3]  # copy / partial / blank, all from step 1
    history: list[dict] = []
    losses: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        batch_loss = 0.0
        for _ in range(args.batch):
            ex = curriculum.next()
            ex["user_input"] = ex["user_input"][: args.max_user]
            ex["answer"] = ex["answer"][: args.max_resp]
            mode = random.choices(EVAL_MODES, weights=mode_weights)[0]
            built = builder.build(ex["user_input"], ex["answer"],
                                  draft_seed_for_mode(ex["answer"], mode))
            delta = model(built["field16"], built["region"], builder.resp_slice)
            logits = model.char_logits(delta, builder.bank_unit)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   built["targets"].reshape(-1))
            (loss / args.batch).backward()
            batch_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(batch_loss / args.batch)
        if step % args.log_every == 0 or step == 1:
            sps = step / max(1e-6, time.time() - t0)
            log("STEP", f"arm={head} step={step}/{args.steps} "
                        f"loss={np.mean(losses[-args.log_every:]):.4f} {sps:.1f}it/s")
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step}
            for mode in EVAL_MODES:
                m = evaluate(model, builder, eval_examples, mode)
                row[mode] = {"exact_fill": m["exact_fill"], "char_acc": m["char_acc"]}
                log(f"EVAL_{mode.upper()}",
                    f"arm={head} step={step} exact_fill={m['exact_fill']:.3f} "
                    f"char_acc={m['char_acc']:.3f} "
                    f"target={m['sample']['target']!r} pred={m['sample']['pred']!r}")
            history.append(row)
    return {"head": head, "params": n_params, "history": history}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--ffn-dim", type=int, default=256)
    ap.add_argument("--max-user", type=int, default=48)
    ap.add_argument("--max-resp", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="perslot,pooled")
    ap.add_argument("--curriculum-dir", default="datasets/recovered/curriculum_v1")
    ap.add_argument("--containers-path", default="")  # off by default: probe speed
    ap.add_argument("--out", default="runs/char_slot_probe/results.json")
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    curriculum = Phase0Curriculum(
        curriculum_dir=args.curriculum_dir,
        containers_path=args.containers_path or None,
        max_chars=args.max_resp,
        history_turns=0,
        rng=random.Random(args.seed),
    )
    eval_examples = []
    for s in curriculum.eval_sentences[: args.eval_n]:
        ex = curriculum._example_from_sentence(s)
        ex["user_input"] = ex["user_input"][: args.max_user]
        ex["answer"] = ex["answer"][: args.max_resp]
        eval_examples.append(ex)

    results = {"args": vars(args), "arms": []}
    for head in [a.strip() for a in args.arms.split(",") if a.strip()]:
        results["arms"].append(run_arm(head, args, curriculum, eval_examples, device))

    outp = pathlib.Path(_ROOT) / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("DONE", f"results -> {outp}")

    # verdict line
    final = {arm["head"]: arm["history"][-1] for arm in results["arms"]}
    for head, row in final.items():
        log("VERDICT", f"{head}: copy={row['copy']['char_acc']:.3f} "
                       f"partial={row['partial']['char_acc']:.3f} "
                       f"blank={row['blank']['char_acc']:.3f} "
                       f"(exact copy={row['copy']['exact_fill']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
