#!/usr/bin/env python3
"""Council runtime engine for Axon (build contract 2026-08-17).

One shared AxonCore (loaded once from a conversational checkpoint) drives N
logical cores. Each core carries its OWN private soul, cloned from the
checkpoint soul and perturbed once at boot with seeded Gaussian noise.

Tick protocol (Jeff's exact spec, see CONTRACT.md):

  Phase A   - every core inhales its private soul, attends the shared field
              (canonical state rendered into the neural view), produces
              delta A, exhales soul.
  Phase B   - every core inhales again, attends the field PLUS all phase-A
              deltas committed into the history region as "\\nCouncil: <delta>"
              lines (the trained pattern), produces refined delta B, exhales.
  Consolidation - the core holding the consolidator role this tick
              (tick % N, round-robin, no permanent consolidator) attends all
              refined deltas (same Council-line rendering) and produces the
              final delta, which commits as the new canonical response_draft.

Ticks continue forever until stopped. A user message may arrive any tick: it
sets user_input and clears response_draft. When the consolidated draft is
unchanged for `stable_ticks` consecutive ticks the turn completes: the draft
appends to conversation_history, user_input clears, and ticking continues
ambient.

Canonical state is all ten doctrine regions as strings. The checkpoint's
trained neural interface is 3 char-slot regions (history 0-127 / user
128-191 / response 192-255), so non-conversational regions enter the neural
view as labeled text inside the history window. Text is NEVER placed in the
response region - the checkpoint was trained with blank drafts (verified OOD
2026-08-17).

Reference: D:/Axon/scripts/multicore_converse.py (proven, read-only).

Identity: Kimmy engine subagent / kimi-k2 / 2026-08-17
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cores.core import AxonCore, CoreConfig  # noqa: E402
from substrate import ALPHABET_SET  # noqa: E402
from training.conversational_objective import decode_parallel_logits  # noqa: E402
from training.trainer_slot import CharSlotFieldBuilder  # noqa: E402

# Doctrine regions (SOURCE_OF_TRUTH: Active Shared Field). Order is fixed.
REGIONS: tuple[str, ...] = (
    "conversation_history",
    "user_input",
    "response_draft",
    "structured_knowledge",
    "situation_awareness",
    "scratch",
    "tool_results",
    "advisor_input",
    "task_state",
    "diary",
)
# Regions with a trained char-slot lane of their own. Everything else renders
# as labeled text inside the history window.
CONVERSATIONAL_REGIONS = frozenset({"conversation_history", "user_input", "response_draft"})

HISTORY_CHARS = 128
USER_CHARS = 64
RESP_CHARS = 64

# Config keys that can be hot-applied without reloading the checkpoint.
HOT_CONFIG_KEYS = frozenset(
    {"tick_delay_ms", "stable_ticks", "max_ticks", "regions", "advisors",
     "log_path", "council_min_conf"}
)

DEFAULT_ADVISOR_TIMEOUT_S = 20.0


def _sanitize(text: str) -> str:
    """Drop characters outside the frozen substrate alphabet (lossy, explicit)."""
    return "".join(ch for ch in text if ch in ALPHABET_SET)


def _region_label(region: str) -> str:
    return region.replace("_", " ").title()


def _mask_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config copy safe to emit: advisor API keys masked to last 4 chars."""
    masked = dict(cfg)
    advisors = []
    for a in cfg.get("advisors") or []:
        a = dict(a)
        key = str(a.get("api_key") or "")
        a["api_key"] = f"...{key[-4:]}" if key else ""
        advisors.append(a)
    masked["advisors"] = advisors
    return masked


class CouncilEngine:
    """Async council tick engine. See CONTRACT.md for the public API."""

    def __init__(self, config_path: Path, event_sink: Callable[[dict], None]):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = ROOT / self.config_path
        self.event_sink = event_sink

        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(self.default_config(), indent=2) + "\n", encoding="utf-8"
            )
        self.config: dict[str, Any] = self.default_config()
        self.config.update(json.loads(self.config_path.read_text(encoding="utf-8")))

        # Canonical state: all ten doctrine regions as strings.
        self.canonical_state: dict[str, str] = {r: "" for r in REGIONS}

        self.running = False
        self.paused = False
        self.tick = 0

        self._core: AxonCore | None = None
        self._builder: CharSlotFieldBuilder | None = None
        self._device: torch.device | None = None
        self._souls: list[torch.Tensor] = []
        self._soul_masks: list[torch.Tensor] = []
        self._temperatures: list[float] | None = None
        self._generators: list[torch.Generator] | None = None
        # Per-core status: soul norm, last delta text, last confidence.
        self._core_status: list[dict[str, Any]] = []

        self._stable = 0
        self._stop_requested = False
        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # Config                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def default_config() -> dict:
        return {
            "checkpoint": "runs/bible_64D_gpu_overnight/ckpt_461500.pt",
            "device": "cpu",
            "cores": 3,
            "soul_noise": 0.05,
            "temperature_spread": 0.0,
            "tick_delay_ms": 250,
            "stable_ticks": 2,
            "max_ticks": 0,
            "council_min_conf": 0.5,
            "regions": {r: True for r in REGIONS},
            "advisors": [],
            "log_path": "",
        }

    def apply_config(self, cfg: dict) -> list[str]:
        """Hot-apply runtime knobs; return keys that need an engine restart."""
        needs_restart: list[str] = []
        for key, value in cfg.items():
            if key not in self.config:
                continue
            changed = self.config[key] != value
            self.config[key] = value
            if changed and key not in HOT_CONFIG_KEYS:
                needs_restart.append(key)
        self.config_path.write_text(
            json.dumps(self.config, indent=2) + "\n", encoding="utf-8"
        )
        return sorted(needs_restart)

    # ------------------------------------------------------------------ #
    # Events                                                              #
    # ------------------------------------------------------------------ #

    def _emit(self, type_: str, **payload: Any) -> None:
        event = {"type": type_, "tick": self.tick, "ts": time.time(), **payload}
        log_path = str(self.config.get("log_path") or "")
        if log_path:
            try:
                p = Path(log_path)
                if not p.is_absolute():
                    p = ROOT / p
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as h:
                    h.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        try:
            self.event_sink(event)
        except Exception:
            pass  # a broken sink must never kill the tick loop

    def _emit_canonical(self) -> None:
        self._emit("canonical", state=dict(self.canonical_state))

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    def raw_config(self) -> dict[str, Any]:
        """The real config, unmasked (server-internal use only)."""
        return dict(self.config)

    def status(self) -> dict:
        n = len(self._core_status)
        return {
            "running": self.running,
            "paused": self.paused,
            "tick": self.tick,
            "consolidator_core": (self.tick % n) if n else 0,
            "cores": [dict(c) for c in self._core_status],
            "canonical_state": dict(self.canonical_state),
            "config": _mask_config(self.config),
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _souls_dir(self) -> Path:
        return Path(__file__).resolve().parent / "souls"

    async def start(self) -> None:
        """Load checkpoint, clone+perturb souls, start the tick loop."""
        if self.running:
            return
        ckpt = Path(str(self.config["checkpoint"]))
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        device = torch.device(str(self.config["device"]))
        self._device = device

        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if "core_state" not in payload or "cfg" not in payload:
            raise ValueError(f"checkpoint {ckpt} missing core_state or cfg")
        cfg = CoreConfig.from_dict(payload["cfg"])
        core = AxonCore(cfg).to(device)
        core.load_state_dict(payload["core_state"])
        core.eval()
        self._core = core
        base_soul = payload["soul_state"]["soul"].to(device)
        base_soul_mask = payload["soul_state"]["soul_mask"].to(device)
        self._builder = CharSlotFieldBuilder(
            HISTORY_CHARS, USER_CHARS, RESP_CHARS, device, torch.float32
        )

        n_cores = int(self.config["cores"])
        soul_noise = float(self.config["soul_noise"])
        self._souls = []
        for i in range(n_cores):
            saved = self._load_saved_soul(i, base_soul.shape, device)
            if saved is not None:
                self._souls.append(saved)
                continue
            g = torch.Generator(device="cpu").manual_seed(1000 + i)
            noise = torch.randn(base_soul.shape, generator=g).to(device, base_soul.dtype)
            self._souls.append(base_soul + noise * float(base_soul.std().item()) * soul_noise)
        self._soul_masks = [base_soul_mask.clone() for _ in range(n_cores)]

        # Optional sampling diversity: per-core temperature + private RNG stream.
        spread = float(self.config["temperature_spread"])
        if spread > 0.0 and n_cores > 1:
            lo, hi = 1.0 - spread, 1.0 + spread
            self._temperatures = [lo + (hi - lo) * i / (n_cores - 1) for i in range(n_cores)]
            self._generators = [
                torch.Generator(device=device).manual_seed(2000 + i) for i in range(n_cores)
            ]
        else:
            self._temperatures = None
            self._generators = None

        self._core_status = [
            {
                "id": i,
                "soul_norm": float(self._souls[i].norm().item()),
                "last_delta": "",
                "last_conf": 0.0,
            }
            for i in range(n_cores)
        ]

        self.running = True
        self.paused = False
        self._stop_requested = False
        self._loop_task = asyncio.create_task(self._loop())
        self._emit("status", **self.status())

    async def stop(self) -> None:
        """Finish the current tick, save per-core souls, halt the loop."""
        if not self.running and self._loop_task is None:
            return
        self._stop_requested = True
        task = self._loop_task
        if task is not None:
            await task  # the loop saves souls in its finally block
        self._emit("status", **self.status())

    def pause(self) -> None:
        self.paused = True
        self._emit("status", **self.status())

    def resume(self) -> None:
        self.paused = False
        self._emit("status", **self.status())

    def _load_saved_soul(
        self, core_id: int, shape: torch.Size, device: torch.device
    ) -> torch.Tensor | None:
        path = self._souls_dir / f"core_{core_id}.pt"
        if not path.exists():
            return None
        try:
            soul = torch.load(path, map_location=device, weights_only=False)
        except Exception:
            return None
        if isinstance(soul, torch.Tensor) and tuple(soul.shape) == tuple(shape):
            return soul.to(device)
        return None

    def _save_souls(self) -> None:
        if not self._souls:
            return
        self._souls_dir.mkdir(parents=True, exist_ok=True)
        for i, soul in enumerate(self._souls):
            torch.save(soul.detach().cpu(), self._souls_dir / f"core_{i}.pt")

    # ------------------------------------------------------------------ #
    # User input                                                          #
    # ------------------------------------------------------------------ #

    async def submit_user_message(self, text: str) -> None:
        """Inject a user message: sets user_input, clears response_draft."""
        text = _sanitize(text.strip())
        if not text:
            return
        self.canonical_state["user_input"] = text
        self.canonical_state["response_draft"] = ""
        self._stable = 0
        self._emit("chat", role="user", text=text)
        self._emit_canonical()

    # ------------------------------------------------------------------ #
    # Neural view rendering                                               #
    # ------------------------------------------------------------------ #

    def _visible(self, region: str) -> bool:
        return bool(self.config.get("regions", {}).get(region, True))

    def _render_history(self, council_lines: list[str] | None = None) -> str:
        """Render canonical state into the history window.

        conversation_history rides as-is; every other non-empty
        non-conversational region enters as labeled text
        (e.g. "\\nScratch: ..."). Council deltas ride as "\\nCouncil: <delta>"
        lines - the trained pattern. The response region stays blank.
        """
        parts = [self.canonical_state["conversation_history"]]
        for region in REGIONS:
            if region in CONVERSATIONAL_REGIONS:
                continue
            content = self.canonical_state[region]
            if content and self._visible(region):
                parts.append(f"\n{_region_label(region)}: {content}")
        for line in council_lines or []:
            parts.append(f"\nCouncil: {line}")
        return _sanitize("".join(parts))[-HISTORY_CHARS * 2 :]

    # ------------------------------------------------------------------ #
    # Core proposal (inhale -> attend -> delta -> exhale)                 #
    # ------------------------------------------------------------------ #

    def _propose(self, core_id: int, history: str, user_input: str) -> tuple[str, float]:
        """One core attends the shared field and produces a delta.

        Inhales its private soul, exhales the updated soul it carries forward.
        Returns (text, confidence) where confidence is the mean probability of
        the produced tokens over non-empty positions.
        """
        assert self._core is not None and self._builder is not None
        core, builder = self._core, self._builder
        soul = self._souls[core_id]
        self._emit("soul", core=core_id, phase="inhale", soul_norm=float(soul.norm().item()))

        built = builder.build(history, user_input, "", "")  # response region stays blank
        temp = self._temperatures[core_id] if self._temperatures else 0.0
        gen = self._generators[core_id] if self._generators else None
        with torch.no_grad():
            out = core.forward_charslot(
                built["field16"],
                built["region"],
                soul,
                soul_mask=self._soul_masks[core_id],
                response_slice=builder.resp_slice,
            )
            logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
            new_soul = out["soul"].detach() if out.get("soul") is not None else soul
            if temp <= 0.0:
                decoded = decode_parallel_logits(logits)
                text = decoded.texts[0]
                probs = torch.softmax(logits, dim=-1)
                top = probs.max(dim=-1)
                nonempty = top.indices[0] != builder.empty_index
                conf = float(top.values[0][nonempty].mean().item()) if nonempty.any() else 0.0
            else:
                probs = torch.softmax(logits[0] / temp, dim=-1)
                if gen is not None:
                    sampled = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                else:
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                chars: list[str] = []
                token_probs: list[float] = []
                for pos, idx in enumerate(sampled.tolist()):
                    if idx == builder.empty_index:
                        break
                    chars.append(builder.bank.chars[idx])
                    token_probs.append(float(probs[pos, idx].item()))
                text = "".join(chars)
                conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.0

        self._souls[core_id] = new_soul  # exhale -> carry -> inhale next phase
        soul_norm = float(new_soul.norm().item())
        self._emit("soul", core=core_id, phase="exhale", soul_norm=soul_norm)
        st = self._core_status[core_id]
        st["soul_norm"] = soul_norm
        st["last_delta"] = text
        st["last_conf"] = conf
        return text, conf

    # ------------------------------------------------------------------ #
    # Advisors (scaffolding: working HTTP path, no training integration)  #
    # ------------------------------------------------------------------ #

    async def _advisor_delta(
        self, advisor: dict, phase: str, council_lines: list[str]
    ) -> dict[str, Any] | None:
        """Query one OpenAI-compatible advisor endpoint; graceful skip on failure."""
        name = str(advisor.get("name") or "advisor")
        result: dict[str, Any] = {"advisor": name, "text": "", "conf": 0.0, "error": None}
        if not advisor.get("enabled") or not advisor.get("api_key"):
            return None
        endpoint = str(advisor.get("endpoint") or "").rstrip("/")
        if not endpoint:
            result["error"] = "missing endpoint"
            self._emit("advisor_delta", **result)
            return result
        state_lines = [f"{_region_label(r)}: {v}" for r, v in self.canonical_state.items() if v]
        if council_lines:
            state_lines.extend(f"Council: {d}" for d in council_lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are advisor "
                    f"{name} in the Axon council (phase {phase}). Read the shared "
                    "field and propose one short response delta as plain text."
                ),
            },
            {"role": "user", "content": "\n".join(state_lines) or "(empty field)"},
        ]
        payload = {
            "model": advisor.get("model") or "gpt-4o-mini",
            "messages": messages,
            "temperature": float(advisor.get("temperature", 0.7)),
        }
        try:
            import httpx

            timeout = float(advisor.get("timeout", DEFAULT_ADVISOR_TIMEOUT_S))
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{endpoint}/chat/completions",
                    headers={"Authorization": f"Bearer {advisor['api_key']}"},
                    json=payload,
                )
                resp.raise_for_status()
                text = str(resp.json()["choices"][0]["message"]["content"]).strip()
            result["text"] = _sanitize(text)
        except Exception as exc:  # graceful skip: network, auth, shape, timeout
            result["error"] = f"{type(exc).__name__}: {exc}"
        self._emit("advisor_delta", **result)
        return result

    async def _gather_advisor_deltas(self, phase: str, council_lines: list[str]) -> list[str]:
        advisors = self.config.get("advisors") or []
        results = await asyncio.gather(
            *(self._advisor_delta(a, phase, council_lines) for a in advisors)
        )
        # Advisor deltas join the pool labeled Advisor:<name>.
        return [f"Advisor:{r['advisor']}: {r['text']}" for r in results if r and r["text"]]

    # ------------------------------------------------------------------ #
    # Tick loop                                                           #
    # ------------------------------------------------------------------ #

    async def _loop(self) -> None:
        try:
            while not self._stop_requested:
                max_ticks = int(self.config["max_ticks"])
                if max_ticks > 0 and self.tick >= max_ticks:
                    break
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._emit("error", message=f"tick {self.tick}: {type(exc).__name__}: {exc}")
                self.tick += 1
                delay = max(0, int(self.config["tick_delay_ms"])) / 1000.0
                if delay:
                    await asyncio.sleep(delay)
        finally:
            self.running = False
            self._save_souls()
            self._loop_task = None
            self._emit("status", **self.status())

    async def _tick(self) -> None:
        """One tick: phase A, phase B, consolidation (contract section 2).

        Every core's delta is produced and emitted every tick (the UI shows
        them all), but only deltas whose confidence passes council_min_conf
        enter the shared field as Council lines. Measured 2026-08-17: raw
        high-temperature deltas committed unfiltered poison the next tick and
        the council diverges instead of converging.
        """
        n = len(self._souls)
        consolidator = self.tick % n
        min_conf = float(self.config.get("council_min_conf", 0.5))
        self._emit("tick_start", consolidator=consolidator)
        user_input = self.canonical_state["user_input"]

        # Phase A - every core attends the shared field, produces delta A.
        history_a = self._render_history()
        deltas_a: list[str] = []
        for i in range(n):
            text, conf = self._propose(i, history_a, user_input)
            self._emit("core_delta", core=i, phase="A", text=text, conf=conf)
            if conf >= min_conf:
                deltas_a.append(text)
        deltas_a.extend(await self._gather_advisor_deltas("A", []))
        await asyncio.sleep(0)  # let the event loop breathe between phases

        # Phase B - every core attends the field PLUS all gate-passing phase-A
        # deltas as "\nCouncil: <delta>" lines, produces refined delta B.
        history_b = self._render_history(deltas_a)
        deltas_b: list[str] = []
        for i in range(n):
            text, conf = self._propose(i, history_b, user_input)
            self._emit("core_delta", core=i, phase="B", text=text, conf=conf)
            if conf >= min_conf:
                deltas_b.append(text)
        deltas_b.extend(await self._gather_advisor_deltas("B", deltas_a))
        await asyncio.sleep(0)

        # Consolidation - core (tick % N) attends all refined deltas and
        # produces the final delta. It commits to response_draft ONLY when it
        # passes the same confidence gate as council lines: a low-confidence
        # consolidation leaves the canonical field untouched (measured
        # 2026-08-17: ungated ambient commits fill the draft with noise).
        history_c = self._render_history(deltas_b)
        final_text, final_conf = self._propose(consolidator, history_c, user_input)
        draft = self.canonical_state["response_draft"]
        if final_conf >= min_conf:
            if final_text == draft:
                self._stable += 1
            else:
                self._stable = 0
                self.canonical_state["response_draft"] = final_text
        else:
            self._stable += 1  # nothing credible to say; field unchanged
        self._emit(
            "consolidated", core=consolidator, text=final_text, conf=final_conf,
            stable=self._stable,
        )
        self._emit_canonical()

        # Stable-draft turn completion: the draft lands in conversation_history,
        # user_input clears, ticking continues ambient. An empty stable draft
        # completes as (silence) rather than committing noise to history.
        if user_input and self._stable >= int(self.config["stable_ticks"]):
            final_text = self.canonical_state["response_draft"]
            if final_text:
                history = self.canonical_state["conversation_history"]
                history = (history + f"\nUser: {user_input}\nAssistant: {final_text}")[
                    -HISTORY_CHARS * 2 :
                ]
                self.canonical_state["conversation_history"] = history
            self.canonical_state["user_input"] = ""
            self._stable = 0
            self._emit("chat", role="axon", text=final_text or "(silence)")
            self._emit_canonical()
