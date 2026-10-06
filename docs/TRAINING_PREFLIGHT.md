# Foundation preflight for Axon Lab

Jeff requires a finished, operable Lab before training starts. This helper is one
backend component of its readiness checklist, not a trainer or a replacement for
that product acceptance requirement. The Lab should call it and display the
results; Jeff should not need a terminal or Python editing to use it.

`tools/training_preflight.py` exposes `run_preflight(root, device, timeout)` and a
JSON CLI for engineering integration. No optimizer steps, model checkpoint
creation, data import, canonical writes, downloads, or service changes occur.
The hardware check executes a tiny tensor operation, not a training run.

Engineering invocation:

    python -B tools/training_preflight.py --device auto

Devices: `auto`, `cpu`, `cuda:0`. Explicit CUDA failure never falls back silently.
Each subprocess is bounded by `--timeout` (default 60 seconds); checks run
sequentially, so this is a per-check timeout rather than an overall deadline.

## Report contract

Schema: `axon-foundation-preflight-v1`.

- `checks`: stable id, plain-language label, pass/fail status, message, details.
- `foundation_passed`: all foundation checks passed.
- `training_authorized`: always false. This helper cannot authorize a run.
- `unverified_requirements`: explicit not-checked boundaries: selected core,
  dataset/split, Heart path, checkpoint resume, backup restore, operator controls.
- Python path/version, requested/selected device, Torch/NumPy versions, actual
  GPU name/VRAM/capability when selected, timestamps and check durations.

Foundation checks cover file presence, both existing substrate self-tests,
native-95 alphabet identity, exact round trips, outsider rejection, the native
bank matching the current sealed lane bank, and a real Torch tensor operation.
They do not repair or certify the previously reported float64 narrowing issue,
Heart admission path, Soul recovery, importer, or character-audit behavior.
The current 16D bank is compared against the existing seal; no vectors or new
codebooks are generated or changed. The existing codec self-tests retain their
current 1024D scope; future widths require their own implementation and tests.

CLI exit 0 means foundation checks passed. It NEVER means training is ready.
Missing dependencies, corrupt banks, failed subprocesses and timeouts yield
failed checks and exit 1. Probe imports use fresh interpreters in the selected
repository and disable bytecode writes. No reports are saved automatically.

## Integration and final acceptance

The UI/Trainer lead owns the actual Start eligibility decision. Combine this
report with real component-contract validation, selected data/split checks,
same-path Heart integration, and restore drills. No manually asserted flag or
mock status should substitute for an executable acceptance test.
Run the probes in a backend worker, keeping browser controls responsive while
checks are pending. A cold Torch import/device initialization can take tens of
seconds; a pending or timed-out probe must not appear as a passed check.

Before Jeff's first training run he must be able to create/configure the core,
choose its curriculum and device, inspect the architecture and states, control
start/pause/resume/stop, see live results, save/select checkpoints, and run
inference from the finished browser interface. Verify phone and desktop behavior
and that control actions affect the real backend. Mid-output resume must restore
the matching exact field/replay boundary without duplicate committed text.
Full builder/cloud/inspection scope is coordinated by the team; this document
does not reduce Jeff's product requirement or freeze an architecture.
