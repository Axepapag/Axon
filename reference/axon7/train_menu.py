#!/usr/bin/env python3
r"""
Axon v7 Training Menu — double-click train.bat to launch.

Plain terminal menu for Jeff's v7 schooling runs. It keeps the front
door focused on the choices that matter most:

  1. RESUME a checkpoint from checkpoints\ and keep schooling it.
  2. NEW CORE — choose d_model, layers, heads, FFN dim, and hot soul slots.

Then choose a curriculum/dataset, total steps, save/log/render cadence,
and whether this is a local run now or a portable cloud command.
Advanced loss weights and internal trainer flags stay on trainer defaults.
"""
from __future__ import annotations

import os
import sys
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import torch  # only used to peek checkpoint metadata in the picker
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Terminal helpers (Windows msvcrt for arrow-key navigation)
# ---------------------------------------------------------------------------

if os.name == "nt":
    import msvcrt

    def get_key() -> str:
        ch = msvcrt.getch()
        if ch in (b"\xe0", b"\x00"):
            ch2 = msvcrt.getch()
            return {b"H": "UP", b"P": "DOWN", b"M": "RIGHT", b"K": "LEFT"}.get(
                ch2, f"SPECIAL:{ch2}")
        try:
            decoded = ch.decode("utf-8")
            if decoded == "\r":
                return "ENTER"
            if decoded == "\x1b":
                return "ESCAPE"
            return decoded
        except UnicodeDecodeError:
            return "?"
else:
    import termios
    import tty

    def get_key() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                if sys.stdin.read(1) == "[":
                    return {"A": "UP", "B": "DOWN", "C": "RIGHT",
                            "D": "LEFT"}.get(sys.stdin.read(1), "ESCAPE")
                return "ESCAPE"
            if ch == "\r":
                return "ENTER"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def menu_select(title: str, options: list[str]) -> str | None:
    """Arrow-key menu. Enter selects, Escape goes back, digits jump."""
    selected = 0
    while True:
        clear_screen()
        print("=" * 64)
        print(f"  {title}")
        print("=" * 64)
        print("  (Up/Down arrows + Enter, or type a number; Esc = back)\n")
        for i, opt in enumerate(options):
            marker = " > " if i == selected else "   "
            print(f"{marker}[{i + 1}] {opt}")
        print()
        key = get_key()
        if key == "UP":
            selected = (selected - 1) % len(options)
        elif key == "DOWN":
            selected = (selected + 1) % len(options)
        elif key == "ENTER":
            return options[selected]
        elif key == "ESCAPE":
            return None
        elif key.isdigit():
            digits = key
            while True:
                k2 = get_key()
                if k2 == "ENTER":
                    break
                if k2.isdigit():
                    digits += k2
                else:
                    break
            idx = int(digits) - 1
            if 0 <= idx < len(options):
                return options[idx]


def prompt_number(label: str, default, min_val=None, max_val=None,
                  is_float: bool = False):
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw) if is_float else int(raw)
            if min_val is not None and val < min_val:
                print(f"  Must be at least {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be at most {max_val}")
                continue
            return val
        except ValueError:
            print("  Please enter a number.")


def prompt_text(label: str, default: str) -> str:
    raw = input(f"  {label} [{default}]: ").strip()
    return raw if raw else default


def prompt_yesno(label: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"  {label} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


# ---------------------------------------------------------------------------
# Checkpoint discovery — schooling checkpoints only
# ---------------------------------------------------------------------------

# Everything is relative to this file, so the whole folder can be zipped
# and run anywhere (Lightning AI, Colab, another box) with no path edits.
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATASET_DIR = ROOT / "datasets"
SIMPLEPOD_HELPER = ROOT / "scripts" / "simplepod_runner.py"
SIMPLEPOD_IDENTITY = Path.home() / ".ssh" / "simplepod_axon_ed25519"


def format_command(cmd: list[Any]) -> str:
    parts = [str(c) for c in cmd]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(p) for p in parts)


def repo_relative_arg(path_text: str) -> tuple[str, bool]:
    """Return a repo-relative path for portable cloud commands."""
    path = Path(path_text)
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except Exception:
        return path_text, False
    return rel.as_posix(), True


def simplepod_trainer_args(cmd: list[str]) -> tuple[list[str], list[str]]:
    """Translate a local trainer command into remote trainer arguments."""
    path_flags = {"--episodes", "--warm-start", "--resume", "--config"}
    args: list[str] = []
    warnings: list[str] = []
    i = 2
    while i < len(cmd):
        item = cmd[i]
        if item == "--save-path" and i + 1 < len(cmd):
            args += [item, "checkpoints"]
            i += 2
            continue
        if item in path_flags and i + 1 < len(cmd):
            rel, ok = repo_relative_arg(cmd[i + 1])
            args += [item, rel]
            if not ok:
                warnings.append(
                    f"{item} points outside {ROOT}; put that file under the repo "
                    "before running on SimplePod."
                )
            i += 2
            continue
        args.append(item)
        i += 1
    return args, warnings


def run_target_label(run_target: str) -> str:
    if run_target == "local":
        return "local run now"
    if run_target == "simplepod_command":
        return "SimplePod Docker command"
    return "future Kaggle/CUDA command"


def default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def suggest_soul_rows(d_model: int) -> int:
    """A sensible default soul size by model width. The soul is a set of
    private memory rows carried tick to tick, each d_model wide. With
    act_reflect_v2 the rows are individually addressable, so serious cores
    should start with a substantial private state."""
    if d_model <= 128:
        return 32
    if d_model <= 256:
        return 64
    if d_model <= 512:
        return 128
    return 256


def suggest_heads(d_model: int) -> int:
    for heads in (8, 4, 2, 1):
        if d_model % heads == 0 and d_model // heads >= 32:
            return heads
    return 1


def suggest_ffn_dim(d_model: int) -> int:
    if d_model <= 128:
        return 16384
    if d_model <= 256:
        return 32768
    if d_model <= 512:
        return 65536
    return max(65536, d_model * 128)


def list_checkpoints() -> list[tuple[str, Path]]:
    """Every .pt under checkpoints\\ that can be resumed by the trainer."""
    found: list[tuple[str, Path]] = []
    if not CHECKPOINT_DIR.exists():
        return found
    for p in sorted(CHECKPOINT_DIR.glob("*.pt"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = p.stat().st_size / 1e6
        when = time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        found.append((f"{p.name:<42s} {size_mb:7.1f}MB  {when}", p))
    return found


def _inspect_checkpoint_header(path: Path) -> dict[str, Any]:
    """Read only the small-metadata keys from a .pt (no tensor materialization).
    Safe to call inside the picker; if the file is malformed or torch isn't
    installed, we return a dict with unknown values instead of crashing."""
    info: dict[str, Any] = {
        "core_id": "?", "step": "?", "kind": "?",
        "size_mb": path.stat().st_size / 1e6,
    }
    if not _HAS_TORCH:
        return info
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        info["core_id"] = str(ck.get("core_id", "?"))
        step = ck.get("step")
        info["step"] = "fresh" if step in (None, 0) else f"step {int(step):,}"
        info["kind"] = str(ck.get("kind", "?"))
    except Exception:
        pass
    return info


def _checkpoint_is_v6(path: Path) -> bool:
    """A v7 core's bank has the region16 head; v6 banks don't."""
    if not _HAS_TORCH:
        return False
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        bank = ck.get("projection_bank_state", {})
        return "ingress.region16.0.weight" not in bank
    except Exception:
        return False


def select_checkpoint() -> Path | None:
    """Pick a .pt. Reads only the small-metadata keys from each file
    to show core_id / step / kind in the row label, no tensor loading."""
    ckpts = list_checkpoints()
    if not ckpts:
        print("\n  No .pt files found in checkpoints\\.")
        input("  Press Enter to go back...")
        return None
    enriched: list[tuple[str, Path, int]] = []
    for label, path in ckpts:
        info = _inspect_checkpoint_header(path)
        row = f"{label}  |  {info['core_id']:<18s}  {info['step']:<12s}  {info['kind']}"
        kind_key = 0 if info["kind"] == "trained" else 1
        enriched.append((row, path, kind_key))
    # Sort trained checkpoints first, preserving newest-first order otherwise.
    enriched.sort(key=lambda item: item[2])
    labels = [label for label, _, _ in enriched]
    choice = menu_select("Resume which checkpoint?", labels)
    if choice is None:
        return None
    return {label: path for label, path, _ in enriched}[choice]


# ---------------------------------------------------------------------------
# New-core configuration — schooling shape
# ---------------------------------------------------------------------------

def new_core_config() -> tuple[str, dict[str, Any]]:
    clear_screen()
    print("=" * 64)
    print("  NEW CORE — schooling shape")
    print("=" * 64)
    print("  Press Enter at any prompt to accept the default.")
    print("  Quiet defaults: dropout=0, ticks=3/2, soul_mode=act_reflect_v2.\n")

    while True:
        core_id = prompt_text("Core name / id",
                              "core_v7_128")
        if not core_id.startswith("core_"):
            core_id = "core_" + core_id
        target = CHECKPOINT_DIR / f"{core_id}.pt"
        if target.exists():
            size_mb = target.stat().st_size / 1e6
            when = time.strftime("%m-%d %H:%M",
                                 time.localtime(target.stat().st_mtime))
            print(f"  WARNING: {target} already exists "
                  f"({size_mb:.1f} MB, last modified {when})")
            if not prompt_yesno("Use this name anyway?", False):
                continue
        break

    cfg: dict[str, Any] = {}
    cfg["d_model"] = prompt_number("Model width d_model", 128, 1)
    cfg["n_layers"] = prompt_number("Transformer layers", 2, 1)
    head_default = suggest_heads(cfg["d_model"])
    while True:
        cfg["n_heads"] = prompt_number(
            "Attention heads (must divide d_model)", head_default, 1)
        if cfg["d_model"] % cfg["n_heads"] == 0:
            break
        valid = [h for h in range(1, 17) if cfg["d_model"] % h == 0]
        print(f"  {cfg['n_heads']} does not divide {cfg['d_model']}. "
              f"Valid: {valid}")
    ffn_default = suggest_ffn_dim(cfg["d_model"])
    print("\n  FFN dim is the core's main inner capacity.")
    print("  Defaults stay fat for schooling without asking for a multiplier.")
    cfg["ffn_dim"] = prompt_number("FFN dimension", ffn_default, 1)
    ffn_params = 3 * cfg["d_model"] * cfg["ffn_dim"] * cfg["n_layers"]
    print(f"  -> ~{ffn_params / 1e6:.0f}M FFN params")
    if cfg["ffn_dim"] > 65536:
        print(f"  NOTE: very large; watch GPU memory (checkpointing helps).")
    soul_default = suggest_soul_rows(cfg["d_model"])
    print(f"\n  Hot soul slots are private {cfg['d_model']}-wide rows carried")
    print("  across ticks. They are the schooling target for memory/soul work.")
    cfg["soul_rows"] = prompt_number(
        "Hot soul slots / soul_rows", soul_default, 0)
    cfg["dropout"] = 0.0
    cfg["n_ticks"] = 3
    cfg["grad_ticks"] = 2
    cfg["soul_mode"] = "act_reflect_v2"
    return core_id, cfg


# ---------------------------------------------------------------------------
# Dataset selection — datasets\ entries or any typed path
# ---------------------------------------------------------------------------

def _browse_directories_arrowkey(start: Path) -> Path | None:
    """Walk folders from `start` using arrow keys. Enter descends into a
    folder or selects a file; the parent (..) entry always appears first;
    Esc cancels. Caps at 200 entries to keep the screen sane; deeper
    subfolders are reached by entering one and re-browsing."""
    current = start
    while True:
        try:
            entries = sorted(current.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
        except (PermissionError, OSError) as e:
            print(f"\n  Cannot read {current}: {e}")
            input("  Press Enter to go back...")
            return None
        labels: list[str] = ["[..]  (parent folder)"] + [
            ("[D] " if p.is_dir() else "[F] ") + p.name + ("\\" if p.is_dir() else "")
            for p in entries
        ]
        if len(labels) > 201:
            labels = labels[:200] + [f"[.. {len(entries) - 200} more, type a path instead)"]
        choice = menu_select(f"Browse: {current}", labels)
        if choice is None:
            return None
        if choice.startswith("[..]"):
            parent = current.parent
            if parent == current:           # at a drive root
                continue
            current = parent
            continue
        if choice.startswith("[.."):
            # "more" sentinel - fall back to typed path
            return _prompt_path_arrowkey(current)
        # map label back to the actual entry
        idx = labels.index(choice) - 1
        picked = entries[idx]
        if picked.is_dir():
            current = picked
            continue
        # file picked - validate it's a trainable type
        if picked.suffix.lower() in (".txt", ".jsonl", ".json", ".csv", ".md"):
            return picked
        print(f"\n  {picked.name} is not a trainable file type.")
        print("  Accepted: .txt, .md, .jsonl, .json, .csv")
        input("  Press Enter to continue browsing...")


def _prompt_path_arrowkey(start: Path | None = None) -> Path | None:
    """Type a Windows path with a sensible default. Drives and UNC paths
    both work. Validates that the path exists and looks trainable."""
    default = str(start) if start else "D:/"
    if not default.replace(":", "").replace("\\", "").replace("/", ""):
        default = "D:/"
    while True:
        raw = prompt_text("Full path to folder or file", default).strip().strip('"')
        if not raw:
            return None
        # Normalize: forward-slashes -> back-slashes for Windows consistency
        path = Path(raw.replace("/", os.sep))
        if not path.exists():
            print(f"  Path does not exist: {path}")
            if not prompt_yesno("Try again?", True):
                return None
            continue
        if path.is_file():
            if path.suffix.lower() not in (".txt", ".jsonl", ".json", ".csv", ".md"):
                print(f"  {path.name} is not a trainable file type.")
                print("  Accepted: .txt, .md, .jsonl, .json, .csv")
                if not prompt_yesno("Use it anyway?", False):
                    continue
            return path
        # Directory: verify it has at least one trainable file somewhere in it
        n_trainable = sum(1 for _ in path.rglob("*")
                          if _.is_file()
                          and _.suffix.lower() in (".txt", ".jsonl", ".json", ".csv", ".md"))
        if n_trainable == 0:
            print(f"  No trainable files under {path}")
            print("  Looking for: .txt, .md, .jsonl, .json, .csv")
            if not prompt_yesno("Use it anyway?", False):
                continue
        print(f"  Found {n_trainable} trainable file(s) under {path}")
        return path


def select_dataset() -> str | None:
    """Pick a dataset. First offers the bundled datasets/ entries, then
    an 'anywhere' option that opens an arrow-key directory browser (or
    lets you type a pasted path with no browsing)."""
    options: list[str] = []
    if DATASET_DIR.exists():
        for d in sorted(DATASET_DIR.iterdir()):
            if d.is_dir():
                n = sum(1 for f in d.rglob("*")
                        if f.is_file()
                        and f.suffix.lower() in (".txt", ".jsonl", ".json", ".csv", ".md"))
                options.append(f"{d.name}\\ ({n} file{'s' if n != 1 else ''})")
        for f in sorted(DATASET_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in (".txt", ".jsonl", ".json", ".csv", ".md"):
                options.append(f.name)
    options.append("<< paste a path from anywhere (or browse folders) >>")
    choice = menu_select("Select curriculum / dataset", options)
    if choice is None:
        return None
    if choice.startswith("<<"):
        picked = _prompt_path_arrowkey()
        return str(picked) if picked is not None else None
    if choice.endswith(")") and " (" in choice:
        # strip the "(N files)" suffix we added in the menu label
        name = choice.rsplit(" (", 1)[0].rstrip("\\")
    else:
        name = choice.rstrip("\\")
    return str(DATASET_DIR / name)


# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

def training_params() -> dict[str, Any] | None:
    clear_screen()
    print("=" * 64)
    print("  SCHOOLING RUN")
    print("=" * 64 + "\n")
    p: dict[str, Any] = {}
    target = menu_select("Where should this run?", [
        "Local: run now on this machine",
        "Future Kaggle/CUDA: print command only",
        "SimplePod Docker GPU: print upload/run command only",
    ])
    if target is None:
        return None
    if target.startswith("Future"):
        p["run_target"] = "kaggle_command"
    elif target.startswith("SimplePod"):
        p["run_target"] = "simplepod_command"
    else:
        p["run_target"] = "local"

    if p["run_target"] == "local":
        dev_default = default_device()   # auto-detects cuda on Lightning AI/Colab
        while True:
            device = prompt_text("Device (cpu / cuda / mps)", dev_default).strip().lower()
            if device in ("cpu", "cuda", "cuda:0", "mps"):
                p["device"] = device
                break
            print("  Accepted values: cpu, cuda, cuda:0, mps")
    elif p["run_target"] == "simplepod_command":
        p["device"] = "cuda"
        print("\n  SimplePod mode does not rent or launch a pod.")
        print("  It prints a helper command you can run after the pod exists.")
        p["simplepod_host"] = prompt_text("SimplePod SSH host", "root@HOST_OR_IP")
        p["simplepod_port"] = prompt_number("SimplePod SSH port (0 = default 22)", 0, 0)
        p["simplepod_remote_dir"] = prompt_text("Remote repo dir", "/workspace/axon7")
        p["simplepod_python"] = prompt_text("Remote Python", "/opt/conda/bin/python")
        identity_default = str(SIMPLEPOD_IDENTITY) if SIMPLEPOD_IDENTITY.exists() else ""
        p["simplepod_identity"] = prompt_text("SSH key path (blank for password)", identity_default)
        print()
    else:
        p["device"] = "cuda"
        print("\n  Kaggle mode does not launch anything from this menu yet.")
        print("  It builds the CUDA command so it can be copied into a future job.\n")

    p["steps"] = prompt_number("Total training steps", 1200, 1)
    p["checkpoint_every"] = prompt_number("Save checkpoint every N steps", 1000, 1)
    p["log_every"] = prompt_number("Display/log every N steps", 25, 1)
    p["render_every"] = prompt_number("Render sample writing every N steps", 100, 1)
    p["name"] = "v7"

    # Quiet trainer defaults. These stay here so the summary/command is explicit,
    # but the menu does not turn them into architecture questions.
    p["lr"] = 0.001
    p["accum"] = 4
    p["weight_decay"] = 0.1
    p["bptt_window"] = 0
    p["ema"] = False
    p["ema_decay"] = 0.999
    p["grad_checkpoint"] = False
    p["deploy"] = False
    p["require_visible"] = False

    if p["device"] == "cpu" and p["steps"] > 5000:
        print("\n  Long CPU runs are usually for deliberate overnight experiments.")
        print("  For big schooling runs, CUDA or a cloud job is usually kinder.")
        if not prompt_yesno("Keep this long CPU run?", False):
            p["steps"] = prompt_number("Training steps", 1200, 1)

    p["runtime_eval_suites"] = []
    print("\n  Final assistant/runtime gates are OFF for early curriculum,")
    print("  state-copy, delta, and soul schooling. Use them as a final exam.")
    if prompt_yesno("Advanced: run final assistant/runtime exam?", False):
        allowed = {"core", "sigils", "repeat", "all"}
        while True:
            raw = prompt_text("Final exam suites", "core,sigils,repeat")
            suites = [s.strip().lower() for s in raw.split(",") if s.strip()]
            bad = [s for s in suites if s not in allowed]
            if suites and not bad:
                p["runtime_eval_suites"] = suites
                break
            print("  Use comma-separated values from: core, sigils, repeat, all")
    return p


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    clear_screen()
    choice = menu_select("AXON v7 SCHOOLHOUSE TRAINER", [
        "Resume a checkpoint from checkpoints\\",
        "New core (choose size, FFN, and hot soul slots)",
    ])
    if choice is None:
        print("\n  Cancelled.")
        return

    resume_path: Path | None = None
    core_id: str | None = None
    cfg: dict[str, Any] | None = None
    warm_start = False

    if choice.startswith("Resume"):
        resume_path = select_checkpoint()
        if resume_path is None:
            print("\n  Cancelled.")
            return
        if _checkpoint_is_v6(resume_path):
            clear_screen()
            print("=" * 64)
            print("  v6 CHECKPOINT DETECTED")
            print("=" * 64)
            print(f"\n  {resume_path.name} is a v6 core. v7 changed the field")
            print("  and head contract, so it cannot be resumed directly.")
            print("\n  WARM-START instead: a fresh v7 core that KEEPS the v6")
            print("  schooling (transformer weights + letter16 heads - its")
            print("  English) and gets a fresh region16 head, fresh optimizer,")
            print("  and a step counter reset to 0.\n")
            if not prompt_yesno("Warm-start this v6 core into v7?", True):
                print("\n  Cancelled.")
                return
            warm_start = True
            default_name = resume_path.stem
            if not default_name.startswith("core_"):
                default_name = "core_" + default_name
            core_id = prompt_text("Core name for the v7 warm start",
                                  default_name + "_v7")
            if not core_id.startswith("core_"):
                core_id = "core_" + core_id
    else:
        core_id, cfg = new_core_config()

    dataset = select_dataset()
    if dataset is None:
        print("\n  Cancelled.")
        return

    params = training_params()
    if params is None:
        print("\n  Cancelled.")
        return

    clear_screen()
    print("=" * 64)
    print("  TRAINING CONFIGURATION")
    print("=" * 64 + "\n")
    if resume_path is not None and warm_start:
        print(f"  Warm-start (v6 -> v7): {resume_path}")
        print(f"  New core:  {core_id}  (keeps v6 English, fresh region head, step 0)")
    elif resume_path is not None:
        print(f"  Resume:    {resume_path}")
    else:
        print(f"  New core:  {core_id}")
        print(f"             d_model={cfg['d_model']} heads={cfg['n_heads']} "
              f"layers={cfg['n_layers']} ffn={cfg['ffn_dim']} "
              f"ticks={cfg['n_ticks']}/{cfg['grad_ticks']} "
              f"soul_rows={cfg['soul_rows']} "
              f"soul_mode={cfg.get('soul_mode', 'concat')}")
    print(f"  Dataset:   {dataset}")
    print(f"  Steps:     {params['steps']:,} x {params['accum']} examples")
    print(f"  Target:    {run_target_label(params['run_target'])}")
    print(f"  Device:    {params['device']}")
    print(f"  Defaults:  lr={params['lr']} accum={params['accum']} "
          f"weight_decay={params['weight_decay']} bptt_window=0 EMA=off")
    print(f"  Final exam: "
          f"{','.join(params['runtime_eval_suites']) if params['runtime_eval_suites'] else 'off'}")
    print(f"  Logs:      progress every {params['log_every']} step(s), "
          f"render every {params['render_every']} step(s)")
    print(f"  Saves:     every {params['checkpoint_every']} steps as "
          "D<model>_heads<heads>_layers<layers>_soul<rows>_<step>steps.pt")
    print("  Deploy:    no (keeps this as a schooling checkpoint)")
    print()
    if not prompt_yesno("Start training?", True):
        print("\n  Cancelled.")
        return

    cmd = [sys.executable, "trainer_v2.py",
           "--episodes", dataset,
           "--steps", str(params["steps"]),
           "--lr", str(params["lr"]),
           "--accum", str(params["accum"]),
           "--weight-decay", str(params["weight_decay"]),
           "--bptt-window", str(params["bptt_window"]),
           "--ema-decay", str(params["ema_decay"]),
           "--checkpoint-every", str(params["checkpoint_every"]),
           "--log-every", str(params["log_every"]),
           "--render-every", str(params["render_every"]),
           "--name", params["name"],
           "--device", params["device"],
           "--save-path", str(CHECKPOINT_DIR)]
    if params["ema"]:
        cmd.append("--ema")
    else:
        cmd.append("--no-ema")
    if params["require_visible"]:
        cmd.append("--require-visible")
    for suite in params["runtime_eval_suites"]:
        cmd += ["--runtime-eval-suite", suite]
    if params["runtime_eval_suites"]:
        cmd += ["--runtime-eval-device", params["device"], "--runtime-eval-check", "pre"]
    if params["grad_checkpoint"]:
        cmd.append("--grad-checkpoint")
    if resume_path is not None and warm_start:
        cmd += ["--warm-start", str(resume_path), "--core-id", core_id]
    elif resume_path is not None:
        cmd += ["--resume", str(resume_path)]
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_json = CHECKPOINT_DIR / f"_{core_id}_{stamp}_config.json"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config_json.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        cmd += ["--config", str(config_json), "--core-id", core_id]
    if not params["deploy"]:
        cmd.append("--no-deploy")

    cmd_str = format_command(cmd)
    if params["run_target"] == "kaggle_command":
        print(f"\n  Kaggle/CUDA command:\n  {cmd_str}\n" + "=" * 64)
        input("  Press Enter to exit...")
        return
    if params["run_target"] == "simplepod_command":
        trainer_args, warnings = simplepod_trainer_args(cmd)
        runner_cmd = [
            sys.executable,
            str(SIMPLEPOD_HELPER),
            "--host",
            params["simplepod_host"],
            "--port",
            str(params["simplepod_port"]),
            "--remote-dir",
            params["simplepod_remote_dir"],
            "--python",
            params["simplepod_python"],
        ]
        if params.get("simplepod_identity"):
            runner_cmd += ["--identity", params["simplepod_identity"]]
        runner_cmd += ["--dry-run", "--"] + trainer_args
        print("\n  SimplePod helper command:")
        print(f"  {format_command(runner_cmd)}")
        print("\n  Remove --dry-run after the host is real and you are ready")
        print("  for the helper to upload, train, and pull checkpoints.")
        if warnings:
            print("\n  Path notes:")
            for warning in warnings:
                print(f"  - {warning}")
        print("\n" + "=" * 64)
        input("  Press Enter to exit...")
        return
    print(f"\n  Running: {cmd_str}\n" + "=" * 64)
    os.system(cmd_str)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Cancelled.")
    except Exception as e:
        print(f"\n\n  Error: {e}")
        input("  Press Enter to exit...")
