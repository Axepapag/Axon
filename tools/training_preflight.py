"""Read-only foundation checks for Axon Lab; never authorizes training."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

SCHEMA = "axon-foundation-preflight-v1"
ROOT = Path(__file__).resolve().parents[1]

SUBSTRATE_PROBE = r'''
import json
import numpy as np
from substrate import native, substrate_1024 as wide
assert native.NATIVE_COUNT == 95 and len(set(native.ALPHABET)) == 95
assert native.ALPHABET == wide.ALPHABET
assert np.array_equal(native.native_bank(), wide.lane_bank()[:95])
sample = native.ALPHABET * 2
assert native.decode_ids(native.encode_ids(sample)) == sample
assert wide.decode_vectors(wide.encode_text(sample)) == sample
for text in ('\t', '\r', '`', '\u00e9', '\U0001f600'):
    for validate in (native.assert_native_text, wide.text_to_ids):
        try:
            validate(text)
        except ValueError:
            pass
        else:
            raise AssertionError('outside-native text accepted')
print(json.dumps({'numpy_version': np.__version__, 'native_count': 95,
                  'native_bank_sha256': native.native_bank_sha256(),
                  'lane_bank_sha256': wide.LANE_BANK_SHA256,
                  'round_trip': True, 'outsider_rejection': True}))
'''

TORCH_PROBE = r'''
import json, sys
import torch
requested = sys.argv[1]
cuda = torch.cuda.is_available()
selected = ('cuda:0' if cuda else 'cpu') if requested == 'auto' else requested
if selected.startswith('cuda') and not cuda:
    raise RuntimeError('CUDA was requested but is unavailable')
device = torch.device(selected)
x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
y = x @ torch.eye(4, dtype=torch.float32, device=device)
if device.type == 'cuda':
    torch.cuda.synchronize(device)
assert torch.equal(x, y)
result = {'torch_version': torch.__version__, 'cuda_available': cuda,
          'requested_device': requested, 'selected_device': str(device),
          'tensor_operation_passed': True, 'training_performed': False}
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(device)
    result.update(device_name=props.name, total_vram_bytes=props.total_memory,
                  compute_capability=[props.major, props.minor])
print(json.dumps(result))
'''

# These are boundaries of this helper, not passes manufactured from declarations.
UNVERIFIED = (
    ("core_contract", "Selected core and component contracts have not been validated."),
    ("dataset_split", "Selected curriculum and held-out split have not been validated."),
    ("heart_runtime", "The selected core's real Heart training/inference path is unverified."),
    ("checkpoint_resume", "Mid-episode and mid-output recovery have not been demonstrated."),
    ("backup_restore", "A separate artifact backup has not been restore-tested."),
    ("operator_controls", "Phone/desktop start, pause, resume, stop and inspection are unverified."),
)


def _check(
    name: str, label: str, command: list[str], root: Path, timeout: float,
    runner: Callable, *, expect_json: bool = False,
) -> dict:
    started = time.monotonic()
    result = {"id": name, "label": label, "status": "fail", "details": {}}
    try:
        completed = runner(command, cwd=str(root), timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", check=False)
        if completed.returncode != 0:
            result["message"] = completed.stderr.strip() or completed.stdout.strip() or (
                f"Check exited with code {completed.returncode}.")
        elif expect_json:
            # A warning may precede the final result. The result must be an object.
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            if not isinstance(payload, dict):
                raise ValueError("probe did not return a JSON object")
            result.update(status="pass", message="Check passed.", details=payload)
        else:
            result.update(status="pass", message="Self-test passed.")
    except subprocess.TimeoutExpired:
        result["message"] = f"Check exceeded {timeout:g} seconds; retry or inspect the environment."
    except (OSError, ValueError, IndexError) as exc:
        result["message"] = f"Check could not complete: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def run_preflight(
    root: Path = ROOT, *, device: str = "auto", timeout: float = 60,
    runner: Callable = subprocess.run,
) -> dict:
    """Return foundation evidence only. Does not mutate a run or launch training.

    Each probe uses an isolated interpreter, avoiding import-cache contamination
    and allowing timeouts. A failed CUDA request never silently falls back to CPU.
    """
    if device not in ("auto", "cpu", "cuda:0"):
        raise ValueError("device must be auto, cpu or cuda:0")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(root).resolve()
    report = {
        "schema": SCHEMA,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": str(root),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "scope": "foundation_only",
        "foundation_passed": False,
        "training_authorized": False,
        "checks": [],
        "unverified_requirements": [
            {"id": name, "status": "not_checked", "message": message}
            for name, message in UNVERIFIED
        ],
    }
    required = ("substrate/substrate.py", "substrate/native.py",
                "substrate/substrate_1024.py", "substrate/d1024_lane_bank.npy")
    missing = [name for name in required if not (root / name).is_file()]
    report["checks"].append({
        "id": "repository", "label": "Current substrate files",
        "status": "fail" if missing else "pass",
        "message": "Required files are missing." if missing else "Required files are present.",
        "details": {"missing": missing},
    })
    if missing:
        return report
    commands = (
        ("substrate16", "Frozen 16D self-test", [sys.executable, "-B", "substrate/substrate.py", "--quiet"], False),
        ("substrate1024", "Sealed 1024D self-test", [sys.executable, "-B", "substrate/substrate_1024.py", "--quiet"], False),
        ("native_boundary", "Native alphabet, sealed bank and round trips", [sys.executable, "-B", "-c", SUBSTRATE_PROBE], True),
        ("device", "PyTorch and a real device tensor operation", [sys.executable, "-B", "-c", TORCH_PROBE, device], True),
    )
    for name, label, command, expect_json in commands:
        report["checks"].append(_check(name, label, command, root, timeout, runner,
                                         expect_json=expect_json))
    report["foundation_passed"] = all(check["status"] == "pass" for check in report["checks"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    report = run_preflight(args.root, device=args.device, timeout=args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Zero means these foundation checks passed, not permission to start a run.
    return 0 if report["foundation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
