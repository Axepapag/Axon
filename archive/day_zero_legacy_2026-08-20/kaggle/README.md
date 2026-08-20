# Axon Phase0b Kaggle Runner

This folder contains the fail-closed Kaggle runner used by the current Axon
charslot continuation pilots.

Build the upload bundle locally:

```powershell
cd D:\Axon
python scripts\build_kaggle_bundle.py
```

Upload `D:\Axon\dist\axon_phase0b_kaggle_bundle.zip` to Kaggle as a dataset, create a GPU notebook, attach that dataset, then run `axon_phase0b_train.ipynb`.

The bundle contains the current 64D checkpoint, the checkpoint-compatible phase0b curriculum, and source needed by `training/trainer_slot.py`. It does not contain raw `D:\00` databases, prior runs, or 8192-slot state artifacts.

The active 128D exact-v3 pilot resumes the valid step-250,000 exact-v2
checkpoint and targets 252,000. Build it with `--leg-kind quarantined_pilot`,
draft weights `0.25,0.35,0.40`, partial/eval fraction `0.50`, LR `2e-4` with
`--resume-lr-policy override`, an audited curriculum reset, weighted family
sampling `0.30,0.40,0.30`, evaluation every 250, full-state checkpoints every
500, and `kaggle/pilot_gate_128d_250000_252000_exact_v3.json`.

The notebook verifies every bundled file, all 95 frozen-substrate characters,
the exact-v3 manifest, source and final checkpoint contracts, identical
source/final fixed-suite hashes, effective LR, optimizer/RNG/sampler state, and
the continuation gate. A passing pilot may emit `continuable: true` but must
always emit `promotable: false`.

When pushing through the Kaggle CLI, request T4 explicitly:

```powershell
kaggle kernels push -p D:\Axon\dist\kaggle_kernel_128D --accelerator NvidiaTeslaT4
```

The default `gpu` accelerator can assign a Tesla P100. On 2026-07-10 Kaggle's Python 3.12/PyTorch image rejected P100 (`sm_60`) because that build supported `sm_70+`, so the trainer failed before initialization.
