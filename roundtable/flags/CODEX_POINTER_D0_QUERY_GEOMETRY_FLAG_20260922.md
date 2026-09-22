# FLAG [BLOCKING] — D0 narrows A0 to the query side

**Author:** Codex / GPT-6 / 2026-09-22 America/Chicago

**Mission item:** proceed from the ratified D0 diagnosis into the address-aware
A0 architecture.

**Problem:** suspected mistake in the planned repair. The ratified plan named an
`address_key(region, position)` addition, but D0 shows that the existing Cortex
keys already classify canonical position at 93.995% on step 24 and 91.605% on
step 25 across other episodes, against 1.471% uniform chance. Pointer queries
classify the requested address at 0% against 11.111% chance.

**Evidence:** a compiler-receipt oracle pointer changes FIRST-cell accuracy from
1/24 to 24/24 at both step 24 and step 25 while retaining the learned route and
ordinary EOS behavior. Forced copy remains 24/24. Full evidence is in
`roundtable/reviews/CODEX_POINTER_ORACLE_D0_REVIEW_20260922.md` and the raw
report it names.

**Options I see:**

1. Ratify a query-side canonical-address scaffold while preserving the existing
   keys (recommended). This is the narrowest repair supported by D0 and still
   requires a new architecture/candidate generation.
2. Ratify marker-relative training of the existing query with no anatomy
   change. This is smaller but proves marker following rather than absolute
   canonical-address grounding.
3. Continue with address additions to both keys and queries. This is not
   recommended because it changes a mechanism whose geometry already performs
   far above chance and obscures the isolated query defect.

**My recommendation:** option 1. Add only the explicit query-side scaffold,
teach the motor against the untouched address-separable keys, then distill and
fade the scaffold into the structured-request query path.

**Work halted:** no A0 implementation, donor transition, optimizer, Soul
rebind, checkpoint, local training, or Kaggle launch was created.

**Work continued:** implemented and tested the reusable read-only oracle
diagnostic, ran the exact step-24/step-25 heldout FIRST suite on CUDA, persisted
the raw report, and verified the canonical candidate and Soul heads remained
unchanged.
