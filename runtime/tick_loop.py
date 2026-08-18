#!/usr/bin/env python3
"""Layer 7 local tick loop.

One tick:
1. Surface dormant knowledge into the active field.
2. Inhale the core's private soul.
3. Attend the char-slot field.
4. Emit response-draft character deltas.
5. Exhale the produced field back into the living soul.

This file is deliberately small runtime plumbing. It proves the breathing loop
and soul persistence; it does not make quality claims about the current cores.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from dataclasses import dataclass

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cores.core import AxonCore, CoreConfig  # noqa: E402
from cores.soul_v2 import SoulManagerV2, SoulState, SoulV2Config, TierSpec  # noqa: E402
from curator.kg_search import KGSearch  # noqa: E402
from training.trainer_slot import CharSlotFieldBuilder, log  # noqa: E402


CONTAINERS = "State/dormant/containers.jsonl"
KG_CACHE = "State/dormant/kg_cache_50k.jsonl"
SOULS_DIR = "State/souls"

DEFAULT_CKPT_PATTERNS = {
    "coreA": [
        "runs/mirror/coreA_charslot_stories/ckpt_*.pt",
        "runs/mirror/coreA_charslot_phase0/ckpt_*.pt",
        "runs/mirror/coreA_phase0/ckpt_*.pt",
    ],
    "coreB": [
        "runs/mirror/coreB_charslot_stories/ckpt_*.pt",
        "runs/mirror/coreB_charslot_phase0/ckpt_*.pt",
        "runs/mirror/coreB_phase0/ckpt_*.pt",
    ],
}


@dataclass(frozen=True)
class RuntimeConfig:
    history_chars: int
    user_chars: int
    response_chars: int

    @property
    def total_slots(self) -> int:
        return self.history_chars + self.user_chars + self.response_chars


def parse_core_list(value: str) -> list[str]:
    cores = [item.strip() for item in value.split(",") if item.strip()]
    if not cores:
        raise ValueError("--cores must name at least one core")
    return cores


def parse_checkpoint_overrides(items: list[str] | None) -> dict[str, pathlib.Path]:
    overrides: dict[str, pathlib.Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--checkpoint expects TAG=PATH")
        tag, raw_path = item.split("=", 1)
        tag = tag.strip()
        if not tag:
            raise ValueError("--checkpoint tag cannot be empty")
        overrides[tag] = pathlib.Path(raw_path.strip())
    return overrides


def _checkpoint_candidates(root: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    return [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix == ".pt" and path.stat().st_size > 0
    ]


def resolve_checkpoint(
    root: pathlib.Path,
    tag: str,
    overrides: dict[str, pathlib.Path] | None = None,
) -> pathlib.Path:
    """Pick the newest complete checkpoint for a core.

    Story checkpoints are preferred by pattern order, with phase0 fallbacks.
    Within one pattern, the newest mtime wins. `.part` files are ignored because
    the mirror can observe an in-progress copy.
    """
    if overrides and tag in overrides:
        path = overrides[tag]
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint override for {tag} not found: {path}")
        return path

    patterns = DEFAULT_CKPT_PATTERNS.get(tag)
    if not patterns:
        raise KeyError(f"no default checkpoint patterns for {tag!r}")
    for pattern in patterns:
        candidates = _checkpoint_candidates(root, pattern)
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    searched = ", ".join(patterns)
    raise FileNotFoundError(f"no complete checkpoint found for {tag}; searched {searched}")


def build_kg_cache(src: pathlib.Path, dst: pathlib.Path, cap: int = 50_000, rebuild: bool = False) -> None:
    """Normalize dormant containers into the word/edge format KGSearch expects."""
    if dst.exists() and not rebuild:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if n >= cap:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            word = (obj.get("word") or obj.get("letters") or obj.get("text") or "").strip()
            if not word:
                continue
            edges = []
            for edge in obj.get("edges", []):
                if isinstance(edge, dict):
                    edge_type = edge.get("edge_type", "")
                    target = edge.get("target", "")
                else:
                    edge_type, target = (list(edge) + ["", ""])[:2]
                if edge_type and target:
                    edges.append([edge_type, target])
            out.write(json.dumps({"word": word, "edges": edges}, ensure_ascii=True) + "\n")
            n += 1
    log("KG", f"cached {n:,} containers -> {dst}")


def render_hits(hits: list[dict], budget: int) -> str:
    """Render surfaced containers as plain sentences in the history region."""
    parts: list[str] = []
    used = 0
    for hit in hits:
        word = hit.get("word", "")
        for edge_type, target in hit.get("edges", [])[:3]:
            sentence = f"{word} {edge_type} {target}."
            if used + len(sentence) + 1 > budget:
                return " ".join(parts)
            parts.append(sentence)
            used += len(sentence) + 1
    return " ".join(parts)


def response_stats(text: str) -> str:
    if not text:
        return "chars=0 uniq=0 top=0.000"
    counts = Counter(text)
    top = max(counts.values()) / max(1, len(text))
    return f"chars={len(text)} uniq={len(counts)} top={top:.3f}"


class TickingCore:
    """One core plus its living soul."""

    def __init__(
        self,
        tag: str,
        ckpt_path: pathlib.Path,
        souls_dir: pathlib.Path,
        device: torch.device,
        runtime_cfg: RuntimeConfig,
        reset_soul: bool = False,
    ):
        self.tag = tag
        self.device = device
        self.ckpt_path = ckpt_path
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = CoreConfig.from_dict(checkpoint["cfg"])
        if not self.cfg.char_slot_mode:
            raise ValueError(f"{tag} checkpoint is not char_slot_mode: {ckpt_path}")
        if runtime_cfg.total_slots > self.cfg.char_slot_max_slots:
            raise ValueError(
                f"{tag} checkpoint supports {self.cfg.char_slot_max_slots} char slots, "
                f"but runtime requested {runtime_cfg.total_slots}"
            )

        self.core = AxonCore(self.cfg).to(device, torch.float32)
        self.core.load_state_dict(checkpoint["core_state"], strict=False)
        self.core.eval()

        soul_cfg = self._soul_config(checkpoint["soul_cfg"])
        self.soul_mgr = SoulManagerV2(soul_cfg, device, torch.float32, n_heads=self.cfg.n_heads)
        self.soul_mgr.load_state_dict(checkpoint["soul_mgr_state"])

        self.soul_path = souls_dir / f"{tag}_soul.pt"
        self.ticks_lived = 0
        if self.soul_path.exists() and not reset_soul:
            saved = torch.load(self.soul_path, map_location=device, weights_only=False)
            self.soul_mgr.state = SoulState.from_saveable(saved["soul_state"], device, torch.float32)
            self.ticks_lived = int(saved.get("ticks_lived", 0))
            log("SOUL", f"{tag}: living soul loaded ({self.ticks_lived} prior ticks)")
        else:
            self.soul_mgr.state = SoulState.from_saveable(checkpoint["soul_state"], device, torch.float32)
            if reset_soul and self.soul_path.exists():
                log("SOUL", f"{tag}: living soul ignored due to --reset-soul")
            else:
                log("SOUL", f"{tag}: soul seeded from training checkpoint")

        self.builder = CharSlotFieldBuilder(
            runtime_cfg.history_chars,
            runtime_cfg.user_chars,
            runtime_cfg.response_chars,
            device,
            torch.float32,
        )

    @staticmethod
    def _soul_config(raw: dict) -> SoulV2Config:
        if hasattr(SoulV2Config, "from_dict"):
            return SoulV2Config.from_dict(raw)
        return SoulV2Config(
            d_model=raw["d_model"],
            tiers=[TierSpec(**tier) if isinstance(tier, dict) else tier for tier in raw["tiers"]],
            categories=raw.get("categories", ["episodic", "lessons", "diary", "awareness", "scratch", "tasks"]),
            router_threshold=raw.get("router_threshold", 0.5),
            write_gate_init=raw.get("write_gate_init", 0.1),
        )

    @torch.no_grad()
    def tick(self, surfaced_text: str, user_input: str) -> str:
        built = self.builder.build(surfaced_text, user_input[: self.builder.user_chars], "", "")
        soul, soul_mask = self.soul_mgr.inhale()
        out = self.core.forward_charslot(
            built["field16"],
            built["region"],
            soul,
            soul_mask=soul_mask.to(self.device),
            response_slice=self.builder.resp_slice,
        )
        logits = self.core.charslot_logits(out["response_delta_16"], self.builder.bank_unit)
        response = self.builder.decode(logits, self.builder.resp_chars).rstrip()
        self.soul_mgr.exhale_after_answer(out["field"])
        self.soul_mgr.maybe_compress()
        self.soul_mgr.maybe_evict()
        self.ticks_lived += 1
        return response

    def save_soul(self) -> None:
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"soul_state": self.soul_mgr.state.to_saveable(), "ticks_lived": self.ticks_lived},
            self.soul_path,
        )
        log("SOUL", f"{self.tag}: soul persisted at {self.ticks_lived} ticks lived")


def scripted_rounds() -> list[tuple[str, str]]:
    return [
        ("jeffrey", "Jeffrey is a"),
        ("axon", "axon is a"),
        ("powershell", "powershell is a"),
        ("memory", "the memory is"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Axon Layer 7 tick loop")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--query", action="append", default=[], help="scripted query; pairs with --input by index")
    parser.add_argument("--input", action="append", default=[], help="scripted user input; pairs with --query by index")
    parser.add_argument("--k", type=int, default=4, help="containers surfaced per tick")
    parser.add_argument("--containers", default=CONTAINERS)
    parser.add_argument("--kg-cache", default=KG_CACHE)
    parser.add_argument("--kg-cap", type=int, default=50_000)
    parser.add_argument("--rebuild-kg-cache", action="store_true")
    parser.add_argument("--souls-dir", default=SOULS_DIR)
    parser.add_argument("--cores", default="coreA,coreB")
    parser.add_argument("--checkpoint", action="append", default=[], help="override checkpoint as TAG=PATH")
    parser.add_argument("--list-checkpoints", action="store_true", help="print resolved checkpoints and exit")
    parser.add_argument("--history-chars", type=int, default=256)
    parser.add_argument("--user-chars", type=int, default=64)
    parser.add_argument("--response-chars", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reset-soul", action="store_true", help="ignore persisted living souls for this run")
    parser.add_argument("--no-persist", action="store_true", help="do not save living souls after ticks")
    parser.add_argument("--dry-run", action="store_true", help="load KG/checkpoint choices but do not load cores")
    args = parser.parse_args(argv)

    root = _ROOT
    tags = parse_core_list(args.cores)
    overrides = parse_checkpoint_overrides(args.checkpoint)
    resolved = {tag: resolve_checkpoint(root, tag, overrides) for tag in tags}

    for tag, path in resolved.items():
        log("CKPT", f"{tag}: {path.relative_to(root)}")
    if args.list_checkpoints or args.dry_run:
        return 0

    runtime_cfg = RuntimeConfig(args.history_chars, args.user_chars, args.response_chars)
    souls_dir = root / args.souls_dir
    device = torch.device(args.device)

    build_kg_cache(
        root / args.containers,
        root / args.kg_cache,
        cap=args.kg_cap,
        rebuild=args.rebuild_kg_cache,
    )
    kg = KGSearch(str(root / args.kg_cache))
    log("KG", f"{len(kg.by_word):,} entities, {len(kg.hub_to_words):,} hubs")

    cores: list[TickingCore] = []
    for tag in tags:
        start = time.time()
        core = TickingCore(tag, resolved[tag], souls_dir, device, runtime_cfg, reset_soul=args.reset_soul)
        cores.append(core)
        log(
            "LOAD",
            f"{tag} ready in {time.time() - start:.1f}s "
            f"({sum(p.numel() for p in core.core.parameters()):,} params)",
        )

    def one_round(query: str, user_input: str) -> None:
        hits = kg.search(query, k=args.k)
        surfaced = render_hits(hits, budget=runtime_cfg.history_chars)
        log("SURFACE", f"query={query!r} -> {len(hits)} hits -> {surfaced[:120]!r}")
        for ticking_core in cores:
            start = time.time()
            response = ticking_core.tick(surfaced, user_input)
            log(
                "TICK",
                f"{ticking_core.tag} #{ticking_core.ticks_lived} ({time.time() - start:.1f}s) "
                f"in={user_input!r} out={response!r} {response_stats(response)}",
            )

    if args.interactive:
        print("query and input per tick; blank query exits.")
        while True:
            try:
                query = input("surface query> ").strip()
                if not query:
                    break
                user_input = input("user input>    ").strip() or query
                one_round(query, user_input)
            except (EOFError, KeyboardInterrupt):
                break
    elif args.query:
        for i, query in enumerate(args.query):
            user_input = args.input[i] if i < len(args.input) else query
            one_round(query, user_input)
    else:
        for query, stem in scripted_rounds():
            one_round(query, stem)

    if args.no_persist:
        log("SOUL", "not persisted (--no-persist)")
    else:
        for ticking_core in cores:
            ticking_core.save_soul()
    log("DONE", f"tick session complete ({', '.join(f'{t.tag}={t.ticks_lived}' for t in cores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
