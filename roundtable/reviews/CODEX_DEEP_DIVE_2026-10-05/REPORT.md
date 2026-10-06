# Current Axon: independent deep dive after Copilot

Codex / GPT-6 / October 5, 2026
Reviewed local HEAD 4fced91, a clean working tree before audit artifacts. Scope: source review, existing tests, substrate checks and isolated synthetic fault probes. No active State, personal database, trained checkpoint, model weights, remote service or architecture was changed.

## Main conclusion

Copilot created a new, smaller foundation at G:/My Drive/Projects/Axon and preserved the former project at G:/My Drive/Projects/Axon_old/Axon_old. The previous broken D64/continuous-port serving assembly is not part of this live repository. The new repository contains tested state/storage/transport primitives. It does not yet contain a working heartbeat application, learned core, trainer, checkpoint manager or GUI.

I independently reran all required checks: 144 tests passed in 12.66 seconds; both substrate self-tests exited 0. These validate the carried/adapted foundation, not trained intelligence, trainer usability, live Heart/Soul transaction coordination or future 1024-reading performance.

Five isolated probes exposed gaps. The most important is a reproducible Soul recovery failure after receipt creation but before HEAD publication. The importer also omits committed WAL-only rows when used with a WAL-backed database. Both need explicit handling before reliable continuation/import is claimed.

## VERIFIED: what exists

| Component | Actual current implementation | Practical limit |
|---|---|---|
| Frozen 16D substrate | Native 95 alphabet, exact cells, no byte transport; invalid canonical text rejected | This is the project's specific 95 characters, not generic printable ASCII: newline is included; backtick, tab, CR, accents and emoji are excluded |
| Frozen 1024 substrate | One vector comprises 64 existing 16-float lane codes; sealed 96x16 bank includes EMPTY padding; exact normal float32 round trips | It is literal lane concatenation, not 95 newly designed 1024-dimensional character codes; zero-tolerance narrowing issue below |
| Shared Field | v4/13 regions, immutable exact text/provenance, masks, deltas, durable branch HEAD | New collaboration region/v5 has not been defined; proposed scope must preserve authority and addresses |
| Heart parts | Authority, valves, queues, durable ingress, lease, masks, identity/turn/health/autobiography helpers | No HeartHost, heartbeat/coordinator, core registry, reasoning barriers or joint field/Soul transaction coordinator |
| Soul | Layered opaque snapshots, prepared transitions, receipts, branch forks, recovery helper | Storage does not implement learned hot-to-cold compression or useful cognitive state; crash case below fails |
| Dormant | Exact experience/source storage, derived lexical/graph index, verified dereference and Cortex materialization | Live State contains README only; no recovered corpus imported; non-native source material cannot be surfaced unchanged |
| Curriculum material | 37 raw files and character audit; reference trainer code preserved | No new lesson schema, converter, sampler, curriculum registry/progress or train/held-out split |
| Core/trainer/application | Direction and handoff documents | No GRU, FFN cycle, mirror-aware learned reader, runtime assembly, checkpoint engine, GUI, cloud bundles or tested backup |

Source map: README.md; docs/CARRY_MANIFEST.md; substrate/native.py; substrate/substrate_1024.py; runtime/field/schema.py; runtime/heart/__init__.py; runtime/soul/store.py; runtime/dormant/evidence_bridge.py. The entire active State file inventory is State/README.md.

## VERIFIED: substrate and carry integrity

- Both substrate self-tests passed independently in this turn.
- Current v7_reference_bank.npy file bytes match the archived old reference exactly (artifact shape 68x16: the original 67 characters plus EMPTY).
- Current d1024_lane_bank.npy bytes also match the archived old artifact exactly.
- native_bank() equals the first 95 rows of the sealed 1024 lane bank exactly. Tests verify the full 95-character native boundary and lane geometry.
- Active runtime/substrate source contains no dependency on the removed unicode_transport, compiler_d64, continuous_d16_port or deleted trainer packages. Historical references remain in docs/history/reference.

The 1024 substrate's mechanics are demonstrably lossless for canonical float32 inputs. That does not prove a GRU reads 64 lanes accurately in one neural step, nor imply 64x throughput. Old 400/400 recall and 76-second training reports describe different historical experiments and were not rerun.

## Reproduced findings, ordered by importance

### F1 [P1] Soul recovery misses receipt-written/HEAD-not-written crashes

Source: runtime/soul/store.py:282-290,319-337,356-373.

finalize_transition persists its receipt before updating HEAD. If saving HEAD then fails, there is a legitimate pending transition with durable receipt and old HEAD. recover treats that receipt as historical advancement and asks for a descendant chain from the new Soul to the old HEAD. It raises SoulIntegrityError: requested soul is not an ancestor of branch HEAD.

Synthetic reproduction: receipt_exists=true; HEAD still old; recover raises; HEAD remains old. All data was under a disposable OS temp directory. The existing tests cover prepared-work recovery but not this crash window.

Recommendation: distinguish HEAD==before, HEAD==after, and genuinely later HEAD. When HEAD is before, finish the recorded receipt-backed transition with its original binding. Add receipt-before-HEAD fault injection and joint field/Soul recovery tests before a trainer's complete-resume promise.

### F2 [P1 for WAL-backed imports] The recovered-memory reader ignores committed WAL rows

Source: curator/import_d00_memories.py:53-67,119-127,136-139,323-330.

The reader opens SQLite with mode=ro&immutable=1. Its snapshot inventory covers named main files, not WAL sidecars, and record extraction reads the original source directory after snapshot copying. This assumes closed, checkpointed, unchanging source databases.

Synthetic reproduction: one committed row lives in an uncheckpointed WAL. A normal read-only SQLite connection sees 1 row; this helper sees 0. The real scattered databases were not opened or modified.

Recommendation: require and verify offline/checkpointed inputs, or create a consistent WAL-aware logical snapshot while separately preserving exact original DB/WAL evidence. Extract from the verified snapshot, not later reads of the original. The importer also expects the old D00 filenames/schema and cannot be assumed suitable for brain.db.

### F3 [P2] 95-character enforcement occurs later than intake admission

Source: runtime/heart/valve.py:189-205,430-478; runtime/field/schema.py:259-266.

A ValveEnvelope containing an emoji is admitted by primitive_valve_registry, while constructing its canonical FieldSpan correctly raises UnsupportedCharacterError. The canonical body remains protected. The intake acceptance contract does not yet enforce the new standing rule at the entry point, and no host exists to coordinate the later failure cleanly.

Recommendation: define native-text validation/rejection before issuing an admission receipt or enqueueing accepted work; retain any rejected external evidence under the explicitly agreed policy. Add front-door cases to the native-substrate tests.

### F4 [P2] Strict vector checks silently narrow precision before validation

Source: substrate/substrate_1024.py:166-179,194-203; analogous conversion in runtime/field/d16_view.py:93-112.

The public API accepts floating input, then casts it to float32 before matching frozen codes. A float64 input changed by 2.220446049250313e-16 still decodes to A at tolerance 0. The altered 16D cell also passes mechanical lifting. That contradicts a literal zero-tolerance input check, although the recovered character is unchanged.

Recommendation: explicitly require the canonical dtype or compare original values before narrowing. Keep the frozen bank unchanged. Add higher-precision input tests instead of relying solely on float32 perturbations.

### F5 [P2] The character audit's CLEAN label is not source or lesson admission proof

Source: tools/audit_characters.py:39-61,64-87,113.

read_text performs newline conversion; splitlines removes CR, vertical-tab and form-feed boundaries. A synthetic TXT source with 4 outside-native controls was reported as 0 outsiders, 5 clean records. Malformed JSONL also falls back to plain text and can be labelled CLEAN.

The docs already say character-clean does not mean lesson-ready. Still, these omissions mean the audit cannot serve as the trainer's format/admission gate. Recommendation: distinguish raw encoding/separator findings from parsed lesson-text findings, flag malformed JSON explicitly, and never silently convert a curriculum. Use a separate versioned conversion tool/report when approved.

All five cases are reproducible with probes.py in this folder. Results are recorded in PROBE_RESULTS.json. No existing runtime code or frozen artifact was edited to produce them.

## VERIFIED: data readiness and duplicate exposure

The combined axon7_curriculum_v2_all.jsonl has 384 records; its eight stage files also total 384, and every record hash overlaps. This is expected because combined and stage files are alternate views of the same lessons. Loading the entire folder indiscriminately would duplicate these examples and could contaminate a random train/held-out split.

The new trainer needs schema conversion from old v7 lesson/state formats, explicit source/episode identities, de-duplication, curriculum versions/progress and episode-level held-out separation. Raw files remain git-ignored because some carry personal journals/conversations. No raw curriculum content was printed or exported in this audit; only hashes/counts were observed.

## Current direction and unresolved contracts

Recorded standing direction: exactly 95 native characters; frozen 16D and 1024D banks; no retired D64 rails, no learned substrate projection, no old continuous-D16 port; Heart-only canonical writes; GRU first; private layered Soul and an inner FFN cycle; GUI trainer sharing runtime's actual state/Heart/Dormant/Soul behavior. Transformers remain possible later. The GRU-first choice is recorded as confirmed by Jeff's measured experience, not merely an assistant preference.

The current 1024 vector literally assembles 64 native cells. The old rail code is absent, but the physical grouping is real. Handoff still flags whether this exact layout matches Jeff's intent. A new independent 1024 codebook must not be silently substituted or the sealed bank regenerated.

Before core/trainer implementation, the read/write contracts still need decisions: lane/region/provenance metadata, 1 versus 64 characters per learned step, exact output selection/termination, the new collaboration region's name/lifetime, frozen-consolidator versus rotated-role semantics, and handling raw non-95 Dormant originals. Layered Soul storage does not itself implement learned compression or prove useful memory.

The inherited WORKING_CONTRACT still references docs/SOURCE_OF_TRUTH.md and Layer 13, while the old doctrine is preserved only in history. Current AGENTS/DIRECTION distinguish new decisions, suggestions and open questions. A concise live restatement would prevent future agents applying retired rail/byte-transport rules or treating unapproved advisor-model suggestions as binding. I have not rewritten doctrine.

## Hardware, backups and delivery state

Independently queried the installed NVIDIA tool: GeForce GTX 1650, 4096 MiB, compute capability 7.5. No neural model was allocated or benchmarked. No promise about Mamba kernels, 64-lane speed or cloud availability is made here.

The new local Git repository has four commits and no configured remote. This new repository has not been pushed through a configured GitHub origin. Google Drive upload/backup completion was not verified. State, checkpoints and raw curricula are intentionally git-ignored, so a future source push would not back up those assets. Backup/recovery must cover them explicitly and be restore-tested.

## Recommendation for the next build

1. Repair/prove Soul crash recovery and define trustworthy text/vector/import admission, without retuning substrate values.
2. Establish one minimal end-to-end Heart loop with exact 95 input, one GRU/core interface, mirror state, durable Soul/field commit and real emitted output. Keep the training State isolated while using the same execution code.
3. Prove copy/recall and restart/continuation through the actual 1024 reader, with a small validated curriculum and measured resources.
4. Build curriculum/version/progress and complete checkpoint/resume/backup around that shared engine. Then expose it through the promised window/buttons.
5. Add optional FFN/multi-core variants as explicit experiments after the baseline and metrics exist. Recorded user choices remain authoritative; these are sequencing recommendations, not a unilateral architecture change.

## VERIFIED / ATTEMPTED / ASSUMED and changes

VERIFIED: current code and docs; 144 passing tests; both substrate self-tests; sealed-artifact equality; five synthetic probe outcomes; 384-record combined/stage overlap; GPU identity; no Git remote; State contains README only.

ATTEMPTED: locating the previously recorded nested checkout failed because Copilot moved/replaced the layout. Review continued at the verified new path.

ASSUMED/NOT VERIFIED: trained core intelligence; GUI/trainer behavior; future 1024 speed; usefulness of Soul compression, FFN or multi-core rotation; remote cloud jobs; private-memory recovery; offsite upload; current Mamba compatibility. Historical metrics were not rerun.

Only review evidence and continuity records were created/updated. No fixes, architecture edits, bank changes, imports of actual private data, training, service launch, package installation, commits or pushes. Tests and probes used disposable directories; no jobs were left running. No external training spend; ordinary account usage was not cost-metered.
