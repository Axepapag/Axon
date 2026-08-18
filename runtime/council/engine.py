#!/usr/bin/env python3
"""Council runtime engine for Axon (build contract 2026-08-17).

Multi-size council: one AxonCore per d_model size (loaded once, shared by
that size's member cores), all attending the SAME canonical 16D field.
Brothers of every width — 64, 128, 256, 512, 1024, 2048, 4096 — stand side
by side; each size's frozen orthogonal rail lifts 16D up to its d_model and
its head projects deltas back down to 16D for exact decode against the
frozen LetterBank. Each member carries its OWN private soul, cloned from its
size's checkpoint soul and perturbed once at boot with seeded Gaussian noise.

Tick protocol (Jeff's spec, amended by Jeff 2026-08-18 — the draft is ALIVE,
nothing is blocked, nothing is stopped early):

  Phase A   - every core inhales its private soul, attends the shared field
              (canonical state + the CURRENT draft rendered as a visible
              "\nDraft: <text>" line), produces delta A, exhales soul.
  Phase B   - every core inhales again, attends the field PLUS all phase-A
              deltas as "\nCouncil: <delta>" lines, produces refined delta B,
              exhales.
  Consolidation - the core holding the consolidator role this tick
              (tick % N, round-robin across ALL cores of ALL sizes — no
              tyrant) attends all refined deltas plus the living draft and
              produces the final delta, which ALWAYS commits as the new
              canonical response_draft. The draft updates every tick. A core's
              gibberish is its own to carry: committed, visible next tick,
              refined tick after tick. Mistakes, adjustment, victory.

Ticks continue forever until stopped. A user message may arrive any tick: the
living draft (if any) commits to conversation_history as "Assistant: <draft>",
then the new user_input is set and the draft restarts for the new turn.
council_min_conf (default 0.0 = OFF) is an experiment lever only: when > 0,
council lines and consolidator commits below the threshold are filtered.

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

# Active window size per region (chars the cores can be shown). The FIELD is
# unbounded; this is only the attended tail. (State/README ruling 2026-07-03:
# active/ = masked-in window, dormant/ = unprocessed region tails.)
ACTIVE_REGION_CHARS = HISTORY_CHARS * 2

# Correct canonical state locations (convener ruling). The full logical field
# materializes under State/active; mask moves and region edits are append-only
# records under State/dormant (masked text preserved byte-for-byte).
ACTIVE_FIELD_PATH = ROOT / "State" / "active" / "council_field.json"
DORMANT_TAILS_PATH = ROOT / "State" / "dormant" / "council_field_tails.jsonl"

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

        # Canonical state: all ten doctrine regions as strings. Region text is
        # NEVER truncated or deleted — the field is the full logical document
        # (runtime/field schema: the window limit belongs to the view, not
        # the field). _masks holds the per-region movable boundary between
        # the attended tail and the dormant prefix:
        #   {"mode": "tail"}    — follow the newest ACTIVE_REGION_CHARS
        #   {"mode": "manual", "offset": N} — Jeff pinned the mask at char N
        self.canonical_state: dict[str, str] = {r: "" for r in REGIONS}
        self._masks: dict[str, dict[str, Any]] = {
            r: {"mode": "tail", "offset": 0} for r in REGIONS
        }
        self._field_path = self._state_path("field_state_path", ACTIVE_FIELD_PATH)
        self._tails_path = self._state_path("dormant_tails_path", DORMANT_TAILS_PATH)
        self._load_field()

        self.running = False
        self.paused = False
        self.tick = 0

        self._builder: CharSlotFieldBuilder | None = None
        self._device: torch.device | None = None
        # Multi-size council: _models holds one loaded AxonCore per d_model;
        # _member_model maps each member core to its model index.
        self._models: list[dict[str, Any]] = []
        self._member_model: list[int] = []
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
            "models": [],
            "soul_noise": 0.05,
            "temperature_spread": 0.0,
            "tick_delay_ms": 250,
            "stable_ticks": 2,
            "max_ticks": 0,
            "council_min_conf": 0.0,
            "regions": {r: True for r in REGIONS},
            "advisors": [],
            "log_path": "",
            "field_state_path": "State/active/council_field.json",
            "dormant_tails_path": "State/dormant/council_field_tails.jsonl",
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
        # State paths take effect immediately (no checkpoint reload). The field
        # reloads ONLY when its path actually changed — the server's config
        # round-trip always includes the key, and reloading on every save
        # would clobber the unsubmitted living draft.
        if "field_state_path" in cfg:
            new_path = self._state_path("field_state_path", ACTIVE_FIELD_PATH)
            if new_path != self._field_path:
                self._field_path = new_path
                self._load_field()
            if "field_state_path" in needs_restart:
                needs_restart.remove("field_state_path")
        if "dormant_tails_path" in cfg:
            self._tails_path = self._state_path("dormant_tails_path", DORMANT_TAILS_PATH)
            if "dormant_tails_path" in needs_restart:
                needs_restart.remove("dormant_tails_path")
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
    # Shared field & dormant state (Jeff's 2026-08-18 ruling)             #
    #                                                                     #
    # The field is the FULL logical document per region — nothing is ever #
    # truncated or deleted. The mask is a movable per-region boundary:    #
    # text behind it is dormant (preserved, inspectable), text past it is #
    # attended by the cores. Mask moves and edits are append-only records #
    # in State/dormant/council_field_tails.jsonl; the live field persists #
    # to State/active/council_field.json (convener ruling 2026-07-03).    #
    # ------------------------------------------------------------------ #

    def _state_path(self, key: str, fallback: Path) -> Path:
        raw = str(self.config.get(key) or "")
        if not raw:
            return fallback
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    def _load_field(self) -> None:
        """Load the persisted field; reset to empty when the file is absent."""
        try:
            saved = json.loads(self._field_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        regions = saved.get("regions") or {}
        masks = saved.get("masks") or {}
        for region in REGIONS:
            text = regions.get(region)
            self.canonical_state[region] = text if isinstance(text, str) else ""
            mask = masks.get(region)
            if isinstance(mask, dict) and mask.get("mode") in ("tail", "manual"):
                self._masks[region] = {
                    "mode": mask["mode"],
                    "offset": max(0, int(mask.get("offset") or 0)),
                }
            else:
                self._masks[region] = {"mode": "tail", "offset": 0}

    def _save_field(self) -> None:
        payload = {
            "regions": self.canonical_state,
            "masks": {
                r: {"mode": m["mode"], "offset": self.mask_offset(r)}
                for r, m in self._masks.items()
            },
        }
        try:
            self._field_path.parent.mkdir(parents=True, exist_ok=True)
            self._field_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _record_tail(self, record: dict[str, Any]) -> None:
        """Append one immutable dormant record (never edit prior lines)."""
        try:
            self._tails_path.parent.mkdir(parents=True, exist_ok=True)
            with self._tails_path.open("a", encoding="utf-8") as h:
                h.write(
                    json.dumps(
                        {"tick": self.tick, "ts": time.time(), **record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError:
            pass

    def mask_offset(self, region: str) -> int:
        """Effective mask boundary: chars [0:offset] dormant, [offset:] active."""
        content = self.canonical_state[region]
        mask = self._masks[region]
        if mask["mode"] == "tail":
            return max(0, len(content) - ACTIVE_REGION_CHARS)
        return max(0, min(int(mask["offset"]), len(content)))

    def _active(self, region: str) -> str:
        """The attended tail of a region (what the cores can be shown)."""
        return self.canonical_state[region][self.mask_offset(region) :]

    def field_view(self) -> dict[str, Any]:
        """Full operator view: every region, its mask, and its dormant part."""
        regions: dict[str, Any] = {}
        for region in REGIONS:
            content = self.canonical_state[region]
            offset = self.mask_offset(region)
            regions[region] = {
                "content": content,
                "dormant": content[:offset],
                "active": content[offset:],
                "mask_offset": offset,
                "mask_mode": self._masks[region]["mode"],
                "total_chars": len(content),
                "dormant_chars": offset,
                "active_chars": len(content) - offset,
                "visible": self._visible(region),
            }
        return {
            "regions": regions,
            "active_region_chars": ACTIVE_REGION_CHARS,
            "model_window_chars": HISTORY_CHARS,
        }

    def set_mask(
        self, region: str, mode: str | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        """Move one region's mask backwards or forwards (Jeff's control)."""
        if region not in self._masks:
            raise ValueError(f"unknown region {region!r}")
        old = self.mask_offset(region)
        mask = self._masks[region]
        if mode is not None:
            if mode not in ("tail", "manual"):
                raise ValueError("mode must be 'tail' or 'manual'")
            mask["mode"] = mode
        if offset is not None:
            mask["mode"] = "manual"
            mask["offset"] = max(0, min(int(offset), len(self.canonical_state[region])))
        new = self.mask_offset(region)
        self._record_tail(
            {
                "type": "mask_move",
                "region": region,
                "old_offset": old,
                "new_offset": new,
                "mode": mask["mode"],
                "masked_text": self.canonical_state[region][:new],
            }
        )
        self._save_field()
        self._emit("field_mask", region=region, old_offset=old, new_offset=new,
                   mode=mask["mode"])
        self._emit_canonical()
        return self.field_view()["regions"][region]

    def set_region(self, region: str, content: str) -> dict[str, Any]:
        """Jeff edits a region. The prior text is preserved in dormant records."""
        if region not in self.canonical_state:
            raise ValueError(f"unknown region {region!r}")
        old = self.canonical_state[region]
        new = _sanitize(str(content))
        if new == old:
            return self.field_view()["regions"][region]
        self._record_tail(
            {
                "type": "region_edit",
                "region": region,
                "old_chars": len(old),
                "new_chars": len(new),
                "superseded_text": old,
            }
        )
        self.canonical_state[region] = new
        self._save_field()
        self._emit("field_edit", region=region, chars=len(new))
        self._emit_canonical()
        return self.field_view()["regions"][region]

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
        """Load every model size, clone+perturb souls per member, start ticking."""
        if self.running:
            return
        device = torch.device(str(self.config["device"]))
        self._device = device

        # Model roster: explicit "models" list, or the legacy single
        # checkpoint+cores pair. Each entry: {"checkpoint": path, "cores": N}.
        models_cfg = self.config.get("models") or [
            {"checkpoint": self.config["checkpoint"], "cores": int(self.config["cores"])}
        ]
        soul_noise = float(self.config["soul_noise"])

        self._models = []
        self._member_model = []
        self._souls = []
        self._soul_masks = []
        model_cache: dict[str, int] = {}
        member_id = 0
        for entry in models_cfg:
            ckpt = Path(str(entry.get("checkpoint") or ""))
            if not ckpt.is_absolute():
                ckpt = ROOT / ckpt
            key = str(ckpt)
            if key not in model_cache:
                model_cache[key] = len(self._models)
                self._models.append(self._load_model(ckpt, device))
            mi = model_cache[key]
            model = self._models[mi]
            base_soul = model["base_soul"]
            base_mask = model["base_soul_mask"]
            for _ in range(int(entry.get("cores", 1))):
                saved = self._load_saved_soul(member_id, tuple(base_soul.shape), device)
                if saved is not None:
                    soul = saved
                else:
                    g = torch.Generator(device="cpu").manual_seed(1000 + member_id)
                    noise = torch.randn(base_soul.shape, generator=g).to(device, base_soul.dtype)
                    soul = base_soul + noise * float(base_soul.std().item()) * soul_noise
                self._member_model.append(mi)
                self._souls.append(soul)
                self._soul_masks.append(base_mask.clone())
                member_id += 1

        n_cores = member_id
        self._builder = CharSlotFieldBuilder(
            HISTORY_CHARS, USER_CHARS, RESP_CHARS, device, torch.float32
        )

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
                "size": self._models[self._member_model[i]]["d_model"],
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

    def _load_model(self, ckpt: Path, device: torch.device) -> dict[str, Any]:
        """Load one AxonCore + base soul from a checkpoint.

        Supports both soul formats: the conversational {soul, soul_mask} pair
        and the charslot_v2 SoulManagerV2 state ({tensor, active, ...}), whose
        rows become the base soul with the active mask.
        """
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if "core_state" not in payload or "cfg" not in payload:
            raise ValueError(f"checkpoint {ckpt} missing core_state or cfg")
        cfg = CoreConfig.from_dict(payload["cfg"])
        core = AxonCore(cfg).to(device)
        core.load_state_dict(payload["core_state"])
        core.eval()
        ss = payload.get("soul_state") or {}
        if "soul" in ss and "soul_mask" in ss:
            base_soul = ss["soul"].to(device)
            base_mask = ss["soul_mask"].to(device)
        elif "tensor" in ss and "active" in ss:
            base_soul = ss["tensor"].unsqueeze(0).to(device)
            base_mask = ss["active"].bool().unsqueeze(0).to(device)
        else:  # no soul in checkpoint: start from a blank active soul
            rows = int(getattr(cfg, "soul_rows", 64)) + int(getattr(cfg, "soul_hot_rows", 64))
            base_soul = torch.zeros((1, rows, cfg.d_model), device=device)
            base_mask = torch.ones((1, rows), dtype=torch.bool, device=device)
        return {
            "core": core,
            "d_model": int(cfg.d_model),
            "checkpoint": str(ckpt),
            "base_soul": base_soul,
            "base_soul_mask": base_mask,
        }

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
        """Inject a user message: the living draft commits to history first.

        Jeff's 2026-08-18 doctrine: a turn ends when Jeff speaks again, never
        by an internal halt. The previous turn's draft (if any) lands in
        conversation_history as "Assistant: <draft>", then the new user_input
        is set and the draft restarts for the new turn.
        """
        text = _sanitize(text.strip())
        if not text:
            return
        prior_draft = self.canonical_state["response_draft"]
        prior_user = self.canonical_state["user_input"]
        # A turn is an EXCHANGE: the draft only commits to history when it
        # was answering a real user message. Drafts from free-ticking before
        # Jeff ever spoke are thinking out loud, not a turn (measured
        # 2026-08-18: phantom "Assistant: <gibberish>" with no User line
        # poisoned the first exchange's context).
        if prior_user:
            history = self.canonical_state["conversation_history"]
            entry = f"\nUser: {prior_user}" if prior_user else ""
            if prior_draft:
                entry += f"\nAssistant: {prior_draft}"
            # The field is NEVER truncated (Jeff 2026-08-18): older turns move
            # behind the region mask into dormant view, they are not deleted.
            self.canonical_state["conversation_history"] = history + entry
            if prior_draft:
                self._emit("chat", role="axon", text=prior_draft)
        self.canonical_state["user_input"] = text
        self.canonical_state["response_draft"] = ""
        self._stable = 0
        self._save_field()
        self._emit("chat", role="user", text=text)
        self._emit_canonical()

    # ------------------------------------------------------------------ #
    # Neural view rendering                                               #
    # ------------------------------------------------------------------ #

    def _visible(self, region: str) -> bool:
        return bool(self.config.get("regions", {}).get(region, True))

    def _render_history(self, council_lines: list[str] | None = None) -> str:
        """Render the ATTENDED view of the field into the history window.

        Only each region's attended tail (past its mask) is rendered — the
        field itself is never truncated. conversation_history's active tail
        rides as-is; every other non-empty non-conversational region enters
        as labeled text (e.g. "\\nScratch: ..."). The council rides as ONE line,
        "\\nCouncil: <strongest full-text proposal>" (_council_lines),
        matching the trained distribution (bible_council_mix.jsonl council
        rows render exactly "Council: <full sentence>" in history);
        per-core fragment digests are off-distribution (measured 2026-08-18).
        The LIVING DRAFT is rendered LAST as "\\nDraft: <text>"
        so tail-truncation always preserves it — every core attends over it
        next tick and refines it. The response region itself stays blank
        (trained layout).
        """
        parts = [self._active("conversation_history")]
        for region in REGIONS:
            if region in CONVERSATIONAL_REGIONS:
                continue
            content = self._active(region)
            if content and self._visible(region):
                parts.append(f"\n{_region_label(region)}: {content}")
        if council_lines:
            parts.append("\nCouncil: " + " | ".join(council_lines))
        draft = self._active("response_draft")
        if draft:
            parts.append(f"\nDraft: {draft}")
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
        assert self._models and self._builder is not None
        model = self._models[self._member_model[core_id]]
        core, builder = model["core"], self._builder
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

    @staticmethod
    def _council_lines(deltas: list[tuple[str, float]]) -> list[str]:
        """Render the field's Council line: the strongest full-text proposal.

        Trained distribution (bible_council_mix.jsonl, 2,200 council rows) is
        a single 'Council: <full sentence>' line in conversation_history.
        Per-core fragments are off-distribution — measured 2026-08-18: 5-char
        fragments starved 10-core rosters, 48-char fragments broke gate 2.
        Every delta is still produced, emitted, and carried every tick; this
        line carries the consensus candidate, it does not filter anyone out.
        """
        texts = [(t, c) for t, c in deltas if t]
        if not texts:
            return []
        best = max(texts, key=lambda tc: tc[1])[0]
        return [best[:48]]

    async def _tick(self) -> None:
        """One tick: phase A, phase B, consolidation (contract section 2).

        Jeff's 2026-08-18 amendment: the draft is ALIVE. Every core's delta is
        produced, emitted, and carried into the shared field every tick; the
        consolidator's delta ALWAYS commits to response_draft. Nothing is
        blocked; nothing stops Axon's thoughts early. council_min_conf > 0 is
        an experiment lever that filters council lines and commits.
        """
        n = len(self._souls)
        consolidator = self.tick % n
        min_conf = float(self.config.get("council_min_conf", 0.0))
        self._emit("tick_start", consolidator=consolidator)
        user_input = self.canonical_state["user_input"]

        # Phase A - every core attends the shared field, produces delta A.
        history_a = self._render_history()
        deltas_a: list[tuple[str, float]] = []
        for i in range(n):
            text, conf = self._propose(i, history_a, user_input)
            self._emit("core_delta", core=i, phase="A", text=text, conf=conf)
            if min_conf <= 0.0 or conf >= min_conf:
                deltas_a.append((text, conf))
        council_a = self._council_lines(deltas_a)
        council_a.extend(await self._gather_advisor_deltas("A", []))
        await asyncio.sleep(0)  # let the event loop breathe between phases

        # Phase B - every core attends the field PLUS the council consensus
        # candidate from phase A, produces refined delta B.
        history_b = self._render_history(council_a)
        deltas_b: list[tuple[str, float]] = []
        for i in range(n):
            text, conf = self._propose(i, history_b, user_input)
            self._emit("core_delta", core=i, phase="B", text=text, conf=conf)
            if min_conf <= 0.0 or conf >= min_conf:
                deltas_b.append((text, conf))
        council_b = self._council_lines(deltas_b)
        council_b.extend(await self._gather_advisor_deltas("B", council_a))
        await asyncio.sleep(0)

        # Consolidation - core (tick % N) attends the refined consensus plus
        # the living draft and produces the final delta, which ALWAYS commits.
        history_c = self._render_history(council_b)
        final_text, final_conf = self._propose(consolidator, history_c, user_input)
        draft = self.canonical_state["response_draft"]
        if min_conf <= 0.0 or final_conf >= min_conf:
            if final_text == draft:
                self._stable += 1
            else:
                self._stable = 0
                self.canonical_state["response_draft"] = final_text
        else:
            self._stable += 1  # gate experiment: field untouched this tick
        self._emit(
            "consolidated", core=consolidator, text=final_text, conf=final_conf,
            stable=self._stable,
        )
        self._emit_canonical()
