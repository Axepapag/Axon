# D64 Receipt-Continuation Local Acceptance

Date: 2026-09-08  
Authority: ratified R1-R12 resolution  
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-08

## Disposition

**PASS for one bounded, private, non-serving Kaggle ablation.**

This is a mechanism acceptance, not evidence that an untrained or trained D64
candidate can reason, converse, terminate correctly, or serve. No checkpoint
was promoted and no serving route was enabled.

## R12 evidence map

| R12 item | Local evidence |
|---|---|
| 1. Native and 2/3/4-cell scalars | `test_receipt_conduit_copies_native_and_two_three_four_cell_scalars` |
| 2. Repeated scalar locations | `test_repeated_scalar_uses_exact_selected_receipt_and_never_crosses_scalar` |
| 3. Changed-source minimal pairs | `test_changed_source_minimal_pair_changes_anchor_identity_but_not_exact_text` |
| 4. Row straddle and page adjacency | Item 1 runs with one-character pages and proves a multi-row four-cell scalar remains exact. |
| 5. False adjacency and joined seams | `test_false_physical_adjacency_and_joined_memory_seams_never_continue` |
| 6. Canonical/proposal collisions | `test_receipts_reject_category_corruption_and_keep_joined_segments_distinct` |
| 7. Wrong receipts/categories | `test_right_categories_under_wrong_receipts_and_provenance_fail_closed` and the joined-segment corruption test |
| 8. Mid-scalar anchor | `test_mid_scalar_anchor_and_substituted_surface_fail_closed` |
| 9. EMPTY/EOS/padding/bounds/mask/provenance/stale identity | The mask/stale-binding, padding/bounds/EOS, provenance, and state-binding tests in `test_d64_pointer_transition.py` |
| 10. Pause/reload/byte-identical resume | Item 1 pauses inside every multi-cell width, serializes, reloads, and compares the complete state byte-for-byte with uninterrupted execution. |
| 11. One causal transition machine | `test_receipt_teacher_path_uses_the_same_causal_recurrent_step` and `test_teacher_scheduled_and_runtime_have_identical_post_scalar_state` |
| 12. Deterministic metric exclusion | `test_continuation_losses_are_masked_while_anchor_and_eos_remain_learned`; evaluation reports continuation integrity separately. |
| 13. EOS co-supervision/retention | `test_receipt_copy_alignment_gate_requires_same_stage_eos_retention` and objective identity assertions |
| 14. Identity change detection | Architecture/objective tests plus `test_state_identity_and_every_runtime_binding_detect_substitution` |
| 15. Legacy-disabled compatibility | `test_disabled_receipt_feature_preserves_legacy_teacher_decoder_path` includes strict state-dict reload and bit-exact output. |
| 16. Complete-field and counterfactuals | Existing field compiler, Unicode, living-reasoning, and canonical-adapter suites; the governed smoke recorded nonzero field/proposal/Soul counterfactuals. |

## Verification

- Focused receipt, objective, and Kaggle catalog tests: **40 passed**.
- Complete 85-test D64/R12 surface: **84 passed** in one run; the sole failure
  was a Windows filename-length error caused by the verifier's temporary-root
  choice. The identical remaining test then **passed** under the short
  `D:\tmp\axr12` root. No tissue change was made to hide that environmental
  failure.
- Targeted Ruff checks passed and `git diff --check` passed. A repository-wide
  Ruff invocation still identifies the pre-existing unsorted `__all__` in
  `training/canonical_d64.py`; that unrelated file was not changed in this
  shot.
- Both Source-of-Truth mirrors were byte-identical before implementation
  acceptance, SHA256
  `c44f16bf4f12d4fe1809313dc843509dc2cf61719db4162903d63f73ee61388c`.

## Governed local mechanism smoke

- Architecture: `living-d64-receipt-823973aed39c1fe14276d2c3`
- Geometry: D64 / 1 head / 4 layers / FFN256
- Parameter count: 331,319
- Candidate generation: `r64v3-16ec1acc9b88381d`
- Effective objective:
  `afc210a59354dee762be63cc8d2adb46678edc413c1533c245ede3412debdf8b`
- Renewable tranche: one optimizer step on the GTX 1650
- Accepted checkpoint:
  `5e81ed60d9d37b4aff2ea3d756e51ef3c082ba6b10e15ba6c70e167949fb8bdb`
- Report:
  `State/training/reasoning/r64v3-16ec1acc9b88381d/segment_000000001_000000001.json`
- Report ID:
  `5e23ad555fcf355a762f0e0f932e19aa4e79db5115058831bcc99de40fe2b706`
- Complete-field coverage: 1.0
- Final counterfactual L2: field 0.148951, proposal 0.327000, Soul 0.473280
- Peak CUDA allocation: 43,270,656 bytes
- Task gate: false; serving promotion claimed: false; complete heldout: false

The zero intelligence metrics after one fresh optimizer step are expected and
remain reported as zero. The smoke proves governed execution, checkpoint/Soul
acceptance, identity isolation, observability, and resource-bounded pause; it
does not claim learning.

## Authorized cloud ablation

The one authorized packet is
`configs/kaggle/axon_d64_mixer_4l_ffn256_h1_receipt_ablation_v1.json`.
It preserves both exact copy-alignment examinations and matches the rejected
Unicode-walk v3 geometry so receipt continuation is the isolated architecture
variable. The large-FFN D64 candidate remains eligible for the later
tournament; this ablation does not select FFN256 as final tissue.

## Remaining serving boundary

Portable full decoder state and byte-exact process-style reload are verified.
Durable ownership of that state by `HeartHost` or another governed runtime
executor is still a serving integration gate. The Kaggle ablation is training
only and cannot promote or serve the candidate.
