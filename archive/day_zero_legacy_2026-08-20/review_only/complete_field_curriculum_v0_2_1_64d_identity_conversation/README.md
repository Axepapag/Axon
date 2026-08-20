# Axon Complete-Field Curriculum Inspection Package v0.2.1

Status: REVIEW ONLY

Author: ChatGPT / GPT-5 / 2026-08-18

The `.1` revision is a focused package revision. Its JSON wire identifiers
remain on the reviewed v0.2 schema family so Codex can compare it directly
with the prior correction contract.

This directory is the corrected inspection package requested after Codex's
v0.1 review. It is deliberately outside `runtime/`, `State/`, `datasets/`, and
`runs/`. Nothing here is imported by the live council.

Jeff's bounded first target is explicit: a 64D identity and conversation core
receives active text from all ten canonical regions through complete ordered
page coverage, then emits one typed transaction updating exactly `scratch`,
`response_draft`, and `diary`. All other content writes remain denied in this
first profile. This restriction is an R0 training choice, not a reversal of
Axon's future field-wide semantic authority.

The package establishes:

- one council tick with proposal, refinement, and consolidation phases;
- exact attributed sibling-delta envelopes and coverage-proven sibling input;
- complete 1, 2, 4, and 8-page counterfactual strata;
- deterministic evidence placement across first, early, middle, late, and
  last eligible positions;
- complete five-variant counterfactual groups;
- separate builder claims and independent verifier evidence;
- default-deny per-core content switches for every canonical region;
- independent per-core attention-mask switches;
- nonempty active text from every canonical region in each focused fixture;
- a default `core-r0-64d-identity-conversation` profile whose successful
  delta contains scratch, response-draft, and diary operations;
- immutable conversation history, user input, tool results, and advisor input;
- current one-boundary masks with a disabled future interval extension point;
- runtime-compatible `shared-field-v1` snapshots and
  `shared-field-delta-v1` payloads inside a versioned council envelope.

`core-v2-governed-full-field` means the policy plane can represent every
canonical region. It does not mean every core is allowed to write every
region. The supplied fixture roster intentionally gives no core full-field
content authority.

## Review commands

From this directory:

```powershell
python -m py_compile generate_curriculum.py verify_curriculum.py tests/test_generate_curriculum.py
python -m unittest discover -s tests -v
python verify_curriculum.py --evidence-out codex_verification_evidence.json
```

To inspect a generated smoke in a new empty directory:

```powershell
python generate_curriculum.py --output-root C:\Temp\axon-v021-smoke --groups 20 --page-size 256
python verify_curriculum.py --generated-root C:\Temp\axon-v021-smoke
```

The builder refuses non-empty output roots. `--groups` counts complete
counterfactual lineages and must be a multiple of 20. The default emits 20
groups and 100 records.

## Prohibitions

This package does not authorize or perform runtime integration, model
training, production shard promotion, live validator changes, or live mask
changes. Codex's independent acceptance is required before R0 reader work.
