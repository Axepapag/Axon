# Kimi Google Organ Demo Runbook — 2026-09-10

Identity stamp: Kimi K2.7 Coding (Kimi Code CLI, bounded demo engineer under Codex) / 2026-09-09
Branch: `codex/kimi-google-demo-20260909` (worktree `D:\Axon-worktrees\kimi-google-demo`)
Operator commands below assume the reviewed runbook has been integrated into
`D:\Axon` on `main`. Commands explicitly labeled VERIFIED were executed on
2026-09-09 on this machine. Nothing here launches training, touches cloud
accounts, changes runtime/training source, or reads private memory text aloud.

---

## 0. One-sentence truth boundary (read this first, verbatim)

**Axon today is a working organism scaffold — an exact 16D Unicode substrate, a
masked Shared Field over one canonical regional state body, a transactional
Heart that is the sole canonical writer, a hash-verified autobiographical memory
of 59,875 recovered records, and an observable spot-native trainer — and it is
NOT a conversational ensemble: no learned Heart or reasoning core is serving,
no general intelligence is claimed, and nothing shown tomorrow is production
deployment.**

## 0.1 The story in one paragraph (presenter framing)

"We build AI tissue the way you'd build a body: first a nervous system that
writes exactly what you tell it (16D substrate, exact Unicode), then attention
that never destroys memory (masked field, dormant-in-place), then a heart that
is the only organ allowed to change the body and that refuses bad input by
design, then autobiographical memory with cryptographic provenance, and finally
a trainer that treats GPU money as renewable tranches with full checkpoint
lineage. The learning organ is real but young: I'll show you exactly what its
latest cloud experiment proved and exactly where it still fails."

## 0.2 Hard rules for the room

- Never say "Axon thinks / understands / is conscious." Say "the Heart
  committed", "the field masked", "the candidate learned".
- Distinguish always: **deterministic mechanism** (substrate, masks, Heart
  transactions, provenance) vs **learned, non-serving** (D64 reasoning core).
- If a live command fails: use the deterministic fallback listed for that
  scene. Never improvise a command on stage.
- Every number quoted on stage is in this runbook with its evidence path.

---

## 1. Preflight checklist (30 minutes before the meeting)

Machine: this Windows box. Open Git Bash. Run all stage commands from `D:\Axon`.

- [ ] **T+0 (2 min)**
      `git status --short --untracked-files=no && git branch --show-current`
      Expect: no tracked changes and branch `main`. Private ignored/untracked
      material is deliberately excluded from the screen.
- [ ] **T+2 (2 min)** Confirm Python: `python --version` → 3.12.x. Set
      `export PYTHONUTF8=1` in every terminal (Windows console is cp1252;
      non-ASCII demo output dies without it).
- [ ] **T+4 (3 min)** Scene 1 dry run: `python substrate/substrate.py` →
      ends with `v7 conformance: PASS` (10 PASS lines). ~0.5 s.
- [ ] **T+7 (2 min)** Scene 1 dry run: Unicode roundtrip one-liner (§2.1)
      → prints `roundtrip exact: True`.
- [ ] **T+9 (2 min)** Scene 2 dry run: mask heredoc (§2.2) → three lines,
      `True` in each.
- [ ] **T+11 (2 min)** Scene 3 dry run:
      `python -m pytest tests/test_heart_control_plane.py -q --basetemp=State/tmp/pytest-demo`
      → `19 passed`. ~10 s. ALWAYS pass `--basetemp=State/tmp/...`: the
      machine's default pytest temp junction is broken (known defect in the
      ledger) and `State/tmp` is the designated scratch area.
- [ ] **T+13 (3 min)** Scene 4 dry run:
      `python -m pytest tests/test_reasoning_circulation.py -q --basetemp=State/tmp/pytest-demo`
      → `5 passed`. ~40 s (torch import dominates; first run may be slower).
- [ ] **T+16 (2 min)** Scene 5 dry run: manifest one-liner (§2.5) →
      `59875`. Also confirm read-only dormant query returns a container.
- [ ] **T+18 (2 min)** Scene 6 dry run:
      `PYTHONDONTWRITEBYTECODE=1 python scripts/axon_kaggle.py --state-root "D:/Axon/State" --json jobs`
      → JSON rows include both receipt jobs with `"phase": "outputs_fetched"`.
- [ ] **T+20 (3 min)** Open evidence files in a text editor for Scenes 5–7 so
      no command needs to run live if time compresses:
      - `D:\Axon\State\training\cloud\jobs\2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a\outputs\axon_job\State\training\trainer\latest_candidate_lifecycle.json`
      - `D:\Axon\roundtable\reports\D64_RECEIPT_KAGGLE_ABLATION_V2_2026-09-09.md`
- [ ] **T+23 (2 min)** Terminal layout per §4; test screen share shows
      non-ASCII (`café 🙂 中文`) correctly.
- [ ] **T+25 (5 min)** Buffer. Do NOT run anything that writes outside
      `State/tmp`. Do NOT open `axon_kaggle_control.ps1` (its menu items
      touch the network or write state).

### If the meeting machine is NOT this machine

Only Scenes 1–4 survive a fresh clone (they are self-contained in the repo).
Scenes 5–6 depend on `D:\Axon\State` evidence. The deterministic fallback for
each scene still works: quote the evidence files, which are quoted verbatim in
this runbook.

---

## 2. Scenes (target 13.5 minutes + slack)

### Scene 1 — The body writes exactly: 16D substrate + Unicode roundtrip (2 min)

- **Purpose:** Prove the lowest layer is exact, frozen, and fail-closed —
  the foundation everything else stands on.
- **Commands (Git Bash, repo root):**
  1. `python substrate/substrate.py`
  2. `PYTHONUTF8=1 python - <<'EOF'` … encode/decode heredoc:
     ```python
     from substrate.unicode_transport import encode_unicode_text, decode_unicode_tokens
     text = "café 🙂 中文"
     toks = encode_unicode_text(text)
     print("tokens:", toks)
     print("roundtrip exact:", decode_unicode_tokens(toks) == text)
     try:
         encode_unicode_text("\ud800")
     except Exception as e:
         print("surrogate rejected:", type(e).__name__, "-", e)
     EOF
     ```
  3. Rejection one-liners (optional if time): decode `(0xC0,)` lead byte and
     an `'a'` spelled as byte `0x61` → `MalformedUnicodeTransportError` /
     `NonCanonicalUnicodeTransportError`.
- **Setup:** none beyond PYTHONUTF8.
- **Expected observable output (verified 2026-09-09):**
  `v7 conformance: PASS` + 10 PASS gates; tokens
  `(2, 0, 5, 290, 264, 62, 335, 254, 248, 225, 62, 323, 279, 268, 325, 245, 230)`;
  `roundtrip exact: True`;
  `surrogate rejected: InvalidUnicodeScalarError - canonical text contains an unpaired surrogate U+D800`.
- **Verified evidence path:** `substrate/substrate.py` (basis, freeze gate
  `verify_substrate`); `substrate/unicode_transport.py` (extended-Hamming
  256-cell byte codebook, IDs 95..350; strict decode). Worst byte-pair cosine
  0.5; all 351 rows unique.
- **What it proves:** canonical text is raw exact Unicode; a native char is one
  frozen 16D cell, every other scalar is strict UTF-8 bytes in typed byte cells;
  malformed input is rejected, not repaired.
- **What it does NOT prove:** nothing about learning. This is a deterministic
  compiler, not a model.
- **Duration:** ~2 min.
- **Fallback:** the all-codepoint test —
  `python -m pytest tests/test_unicode_transport.py -k "not permanent_d64_reader" --basetemp=State/tmp/pytest-demo`
  (22 passed, ~29 s; loops all 1,112,064 valid codepoints). If pytest fails
  entirely, show `substrate/substrate.py` output already captured in §0.2 of
  this file and move on.

### Scene 2 — Attention without amnesia: per-region masks, dormant-in-place (1.5 min)

- **Purpose:** Show the Shared Field is a *view* over one canonical body;
  masking changes attention, never memory.
- **Command:**
  ```python
  PYTHONUTF8=1 python - <<'EOF'
  from runtime.field import LogicalRegion, RegionMaskPolicy, RegionState, SharedFieldSnapshot, D64FieldCompiler
  history = RegionState.from_text(LogicalRegion.CONVERSATION_HISTORY, "abcdefghij",
                                  span_id="first-to-latest", provenance="lived-history")
  field = SharedFieldSnapshot(tick_id=9, regions=(history,))
  c = D64FieldCompiler()
  for pct in (0, 50, 100):
      comp = c.compile(field, region_masks={LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("tail_percent", pct)})
      print(pct, repr(comp.region_text(LogicalRegion.CONVERSATION_HISTORY)),
            comp.source_field_id == field.field_id, repr(field.region(LogicalRegion.CONVERSATION_HISTORY).text))
  EOF
  ```
- **Expected observable output (verified):**
  `0 '' True 'abcdefghij'` / `50 'fghij' True 'abcdefghij'` /
  `100 'abcdefghij' True 'abcdefghij'`.
- **Verified evidence path:** `runtime/field/schema.py`
  (`RegionMaskPolicy`, `resolve_mask_policy`), `runtime/field/compiler_d64.py`
  (`D64FieldCompiler.compile`), `runtime/heart/masks.py` (Heart-owned durable
  slider), doctrine `docs/SOURCE_OF_TRUTH.md` ("masked partition is dormant in
  place"). Tests: `tests/test_heart_region_masks.py` (mask movement preserves
  cells, addresses, `field_id`; identity region unmaskable).
- **What it proves:** masked characters remain in Dormant state in the same
  canonical body; moving the attention slider produces a new rail *view* over
  the same `field_id`. No recall/rematerialization, no deletion.
- **What it does NOT prove:** it does not prove anything about *which* regions
  a learned core would choose to attend; the slider here is operator-set.
- **Duration:** ~1.5 min.
- **Fallback:** `python -m pytest tests/test_heart_region_masks.py tests/test_substrate_contract.py -q --basetemp=State/tmp/pytest-demo` (18 passed, ~16 s). If all else fails, narrate the three-line verified output above.

### Scene 3 — The Heart is the only writer: one commit, one fail-closed rejection (3 min)

- **Purpose:** This is the governance centerpiece. Show a valid typed proposal
  committing transactionally, and bad input being refused with no state change.
- **Command:**
  `python -m pytest tests/test_heart_control_plane.py -q --basetemp=State/tmp/pytest-demo`
  → `19 passed` in ~10 s.
- **Live narration while it runs** (the test names are the demo; run with
  `-v` if Jeff wants to read them):
  - `test_validated_consolidator_decision_commits_via_canonical_delta_path` — a
    typed `FieldDelta` proposal, both board barriers closed, one
    `HeartTransactionBoundary.commit` → `HeartCommit` receipt with `commit_id`;
    successor state roundtrips through the D64 compiler; a second commit from
    the same tick fails closed (`FinalCommitAlreadyMadeError`).
  - `test_core_grants_are_never_commit_capable` — a reasoning-core authority
    trying to commit directly raises `CoreCommitError`. Cores propose; only
    the consolidator path through Heart commits.
  - `test_boundary_rejects_conflicting_sparse_edits` — a malformed delta
    (overlapping edits) raises `OverlappingDeltaError`; the tick stays in
    flight, no partial write.
  - `test_invalid_grants_fail_closed` — six invalid authority grants rejected
    at construction.
- **Setup:** none (no torch in this test file; fast).
- **Expected observable output:** `................... [100%]` + `19 passed`.
- **Verified evidence path:** `runtime/heart/transaction.py`
  (`HeartTransactionBoundary.commit`), `runtime/heart/authority.py`
  (`AuthorityGrant` fail-closed `__post_init__`), `runtime/heart/host.py`
  (submit never constructs grants, never touches canonical state),
  `tests/test_heart_host.py` (sole-writer lease: a second `HeartHost` on the
  same state root gets `LeaseDeniedError`).
- **What it proves:** exactly one writer; typed proposals; transactional
  commit; fail-closed rejection (exception, no receipt, no mutation); tick
  ends only at a Heart commit.
- **What it does NOT prove:** no learned judgment is involved — the
  consolidator decision in the test is a fixture, not a serving model.
- **Duration:** ~3 min.
- **Fallback:** if pytest breaks, run the same three tests individually:
  `python -m pytest tests/test_heart_control_plane.py::test_validated_consolidator_decision_commits_via_canonical_delta_path tests/test_heart_control_plane.py::test_core_grants_are_never_commit_capable tests/test_heart_control_plane.py::test_boundary_rejects_conflicting_sparse_edits -q --basetemp=State/tmp/pytest-demo`.

### Scene 4 — Reasoning rail + proposal workspace: where cores would argue (2 min)

- **Purpose:** Show the D64 rail the Shared Field compiles to, the two-barrier
  proposal board (a noncanonical workspace), and malformed core output being
  rejected but accounted.
- **Command:**
  `python -m pytest tests/test_reasoning_circulation.py -q --basetemp=State/tmp/pytest-demo`
  → `5 passed` (~40 s; torch import dominates — start it talking).
- **What to narrate:**
  - `test_host_runs_both_barriers_finalizes_turn_and_deposits_loadable_episode`:
    user ingress `"hello λ🧠"` → one heartbeat runs first-pass barrier,
    refinement barrier, consolidator commit; asserts the D64 rail carries the
    exact workspace text (`require_rail(64).text == readable_text()`); the
    response lands in `RESPONSE_DRAFT`; tick closes cleanly.
  - `test_malformed_core_output_is_rejected_but_accounted_at_both_barriers`:
    a core emitting with the wrong `author_core_id` is rejected
    (`ParticipantState.FAILED`, "author differs") — the turn still completes
    and canonical state stays clean.
  - `test_board_is_a_noncanonical_workspace`: board activity never changes
    `field_id`, `tick_id`, or `canonical_hash`.
- **Say explicitly:** "This is the *machinery* of a reasoning ensemble — rail,
  workspace, barriers, accounting. The learned cores that would fill it are
  not serving; what you saw are fixture cores proving the plumbing is exact."
- **Verified evidence path:** `runtime/heart/board.py` (`ProposalBoard`,
  `close_first_pass`/`close_refinement`), `runtime/heart/proposal_workspace.py`
  (`D64ProposalWorkspaceRenderer`), `runtime/field/compiler_d64.py`.
- **What it proves:** exact rendering into the current D64 home rail; a
  separately tested width-generic proposal-workspace serializer can render D64
  and D128 workspace rows without claiming a D128 canonical field compiler;
  two-stage barrier governance, rejection-with-accounting, and clean tick
  lifecycle.
- **What it does NOT prove:** no real learned core participated; no
  conversation is being generated.
- **Duration:** ~2 min.
- **Fallback:** the pure-barrier tests without torch:
  `python -m pytest tests/test_heart_control_plane.py::test_board_barriers_and_participant_accounting tests/test_heart_control_plane.py::test_board_is_a_noncanonical_workspace -q --basetemp=State/tmp/pytest-demo`. If everything fails, show this scene as the recorded `5 passed` line captured in preflight.

### Scene 5 — Autobiographical memory: 59,875 records, cryptographic provenance (2 min)

- **Purpose:** Show lived-experience memory recovered from the project's own
  history, with per-record provenance to the original database row, and
  retrieval measured live.
- **Setup:** read-only access to `D:\Axon\State` evidence. All queries use
  `mode=ro&immutable=1` SQLite URIs — nothing is written.
- **Commands:**
  1. Count from the content-addressed import manifest:
     `python -c "import json,pathlib; m=json.loads(pathlib.Path(r'D:\Axon\State\dormant\experience_v1\imports\718f33bf470b90f3f1b2375de3aeb8bf4f47f48c5440b21395f9a2d0b3933ba6\manifest.json').read_text(encoding='utf-8')); print(m['record_count']); print(m['record_kind_counts'])"`
     → `59875` + kind breakdown (28,410 messages; 20,407 episodes; 8,567
     backlog jobs; …).
  2. Timed provenance-carrying retrieval (verified 2026-09-09 twice: seek +
     SHA-256 verify of one container in **0.40–1.30 ms**; term/index lookup
     ~2 ms):
     ```python
     PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 python - <<'EOF'
     import json, time, sqlite3, hashlib
     from pathlib import Path
     idx = Path(r"D:\Axon\State\dormant\.derived\evidence_v1\index.sqlite3")
     con = sqlite3.connect(idx.as_uri()+"?mode=ro&immutable=1", uri=True)
     con.row_factory = sqlite3.Row
     print(dict(con.execute("SELECT key,value FROM meta").fetchall()))
     row = con.execute("SELECT container_id, kind, byte_offset, byte_length, raw_sha256 FROM containers WHERE kind='episode' LIMIT 1").fetchone()
     t0 = time.perf_counter()
     with Path(r"D:\Axon\State\dormant\containers.jsonl").open("rb") as raw:
         raw.seek(row["byte_offset"]); line = raw.read(row["byte_length"])
     verified = hashlib.sha256(line).digest() == row["raw_sha256"]
     rec = json.loads(line)
     t1 = time.perf_counter()
     print(f"seek+stored-hash verify: {(t1-t0)*1000:.2f} ms")
     print("verified:", verified, "| id:", row["container_id"], "| kind:", row["kind"])
     print("source:", rec.get("source"), "| provenance:", rec.get("provenance"))
     print("private text bytes withheld:", len((rec.get("text") or "").encode("utf-8")))
     EOF
     ```
     → `verified: True` for container `c-a166402d7bbb`, provenance
     `recovered_corpus_builder:episode_import:v1`, source
     `D:\00\axon_episodic_memory.db:episodes:1`; the private record text is not
     printed.
- **Verified evidence path:** `State/dormant/experience_v1/imports/718f33bf…/records.jsonl`
  (770 MB, hash-chained, manifest count == line count), byte-exact
  `source_snapshots/ed947737…/` of the seven original `D:\00` sources,
  derived 4.4 GB SQLite index (427,001 containers / 351,978 edges).
  Builder/importer: `curator/import_d00_memories.py`, `curator/recovered_corpus_builder.py`.
- **What it proves:** the selected real episode record is byte-addressable,
  matches the hash stored in the derived index, and traces to a source DB row;
  the corpus manifest and import records preserve source/provenance identities.
- **What it does NOT prove:** the full integrity walk takes ~103 s (SHA-256
  over 770 MB + 3.5 GB), not sub-second — sub-second is lookup only. Retrieval
  shown is key/index lookup, not semantic recall by a learned model.
- **Duration:** ~2 min.
- **Fallback:** open `manifest.json` in an editor and read the count; quote
  the verified 0.40–1.30 ms retrieval numbers from this runbook. Do NOT run the
  ~103 s full verify on stage.

### Scene 6 — Trainer observability: lineage, renewable pause, private job visibility (2 min)

- **Purpose:** Show the training organ is operated like infrastructure:
  content-addressed lineage, pause-at-checkpoint as a designed state, and a
  local window into private Kaggle jobs — **without launching anything**.
- **Commands (Git Bash):** first set
  `V2=/d/Axon/State/training/cloud/jobs/2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a`.
  1. `cat "$V2/outputs/axon_job/State/training/trainer/latest_candidate_lifecycle.json"`
     → `"status": "paused"`, `"step": 120`,
     `"reason": "execution segment ended at an exact accepted checkpoint; curriculum stage remains open for a later renewable tranche"`.
  2. `cat "$V2/outputs/axon_job/State/training/trainer/candidates/reasoning-d64-axon-d64-mixer-4l-ffn256-h1-receipt-v1/r64v3-a80504169350a5ce/accepted_steps/pointer.json"`
     → accepted-step pointer with `current_step: 120`, rolling bundle ids
     (content hashes). Pair with `checkpoint_done.json` sentinel and
     `latest_checkpoint.json` (`optimizer_included: true`,
     `gradient_state_included: true` — resumable, not just weights).
  3. Local job catalog (offline, no network):
     `PYTHONDONTWRITEBYTECODE=1 python scripts/axon_kaggle.py --state-root "D:/Axon/State" --json jobs`
     → both receipt jobs listed with `"phase": "outputs_fetched"` (verified).
     Use `--json` — the interactive picker is not stage-friendly.
  4. Optional if asked about the live dashboard:
     `PYTHONDONTWRITEBYTECODE=1 python scripts/axon_training_watch.py 2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a --local --replay --qa`
     (fully offline replay of the fetched run's 126 events; Ctrl+C to exit).
- **Verified evidence path:** `runtime/trainer/tranche.py` (tranche =
  bounded renewable execution allowance), `runtime/trainer/lifecycle.py`
  (`CandidateStatus`), job bundles under `D:\Axon\State\training\cloud\jobs\`.
- **What it proves:** every checkpoint/pointer filename is its content hash;
  pause is a first-class durable state at an exact accepted step; resume state
  (optimizer + gradients) is preserved; job history is inspectable offline.
- **What it does NOT prove:** no live Kaggle status is queried (offline
  catalog shows locally known phases; 5 old jobs show stale `SUBMITTED`
  because they were never fetched — say "local truth" if asked). No new job
  is launched or resumed.
- **Duration:** ~2 min.
- **Fallback:** the two `cat` commands plus the printed `jobs` rows quoted in
  this runbook. Never fall back to `status`/`monitor`/`doctor` subcommands —
  they hit the network.

### Scene 7 — The learning organ, honestly: D64 receipt result (2 min, mostly slide)

- **Purpose:** Present the latest research result with the exact numbers and
  the exact failure. This is the credibility scene — volunteer the bad numbers
  before anyone asks.
- **Material:** `roundtable/reports/D64_RECEIPT_KAGGLE_ABLATION_2026-09-09.md`
  (v1) and `roundtable/reports/D64_RECEIPT_KAGGLE_ABLATION_V2_2026-09-09.md`
  (v2). Two controlled cloud ablations, same architecture (D64, 1 head,
  4 layers, FFN256, 331,319 parameters), same 120-step tranche, different
  objective weights. No live command needed; the evidence is the fetched,
  rehashed job bundles on disk.
- **The table to show (all values from the adjudication reports):**

  | Metric | Unicode-walk v3 | Receipt v1 (`4b228ccd…`) | Receipt v2 (`2f5687a6…`) |
  |---|---:|---:|---:|
  | Heldout/regression source-position accuracy | 0.645/0.657 | **1.000/1.000** | **1.000/1.000** |
  | Heldout/regression content-only (teacher-forced) | 0.645/0.657 | **1.000/1.000** | **1.000/1.000** |
  | Copy-route gate | 1.000 | 0.000 | **1.000** |
  | EOS-route gate | 0.000 | **1.000** | 0.000 |
  | Emitted EOS token (heldout / regression) | 0.000 | 0.125 / 0.208 | 0.125 / 0.333 |
  | Teacher-forced token accuracy | 0.364 | 0.475 | **0.475** |
  | Exact terminated payload transport | 0.000 | 0.333 | **0.000** |
  | Heldout final mean loss | 0.5760 | 0.7322 | **0.4461** |

- **The honest sentence:** "The receipt mechanism fixed what it was designed
  to fix — exact anchor position and exact supervised content at 1.000 across
  heldout and regression. The learned gate is now the bottleneck: v2 always
  picks COPY, never EOS, so exact *terminated* payload transport is 0.000 and
  teacher-forced token accuracy is 0.475. We have a hypothesis (scalar-bias
  dominance; checkpoint inspection shows `copy_gate.weight` norm 0.094,
  bias −0.0098) and a ratified, observation-only diagnostic plan — no
  architecture change before this meeting."
- **What it proves:** deterministic receipt continuation removed the observed
  multi-cell position/content-only failure on these examinations. The remaining
  learned gate/termination failure is localized, measured, and reproducible
  across two controlled runs. It does not establish that the whole learned
  architecture is sound.
- **What it does NOT prove:** no serving readiness, no end-to-end learned
  transport, no architecture-tournament conclusion. Both candidates are
  paused, renewable, non-serving; both task gates failed.
- **Duration:** ~2 min.
- **Fallback:** the reports themselves. Nothing can "fail live" in this scene.

---

## 3. Launch sequence

There is no single launcher; the honest sequence is per-scene. Minimal stage
set (in order):

```bash
cd /d/Axon
export PYTHONUTF8=1
python substrate/substrate.py                                   # Scene 1
# (Scene 1 heredoc + Scene 2 heredoc as above)
python -m pytest tests/test_heart_control_plane.py -q --basetemp=State/tmp/pytest-demo        # Scene 3
python -m pytest tests/test_reasoning_circulation.py -q --basetemp=State/tmp/pytest-demo      # Scene 4 (start, narrate Scenes 5-6 while it runs)
# Scene 5 dormant one-liners; Scene 6 cat + jobs --json; Scene 7 slide
```

## 4. Terminal layout suggestion for Jeff

- **Terminal A (left, shared):** substrate + field demos (Scenes 1–2). Large
  font; this one shows non-ASCII, so verify `PYTHONUTF8=1` first.
- **Terminal B (right, shared):** pytest runs (Scenes 3–4). Run Scene 4 first
  in the background of the narrative while Scenes 5–6 happen in Terminal C,
  since Scene 4's torch import takes ~40 s.
- **Terminal C (presenter only):** evidence cats and dormant one-liners
  (Scenes 5–6) — or have these pre-opened in an editor as files.
- **Slide deck:** Scene 7 table + truth-boundary slide + the ask (§6).

## 5. Anticipated Google engineering questions — candid answers

- **"Is it conscious / does it understand?"** "No, and we don't claim it. What
  you saw are deterministic organs plus one small non-serving learned core.
  The Heart's authority model is engineered governance, not sentience."
- **"Why not just use a big LLM for the writer?"** "Because we need a single
  accountable writer with transactional semantics and fail-closed rejection.
  An LLM inside would be a proposal source, never the committer — that
  separation is the design."
- **"The 1.000 numbers — is the model solved?"** "No. 1.000 is route, position,
  and teacher-forced content-only alignment. Free-running exact terminated
  transport is 0.000 and teacher-forced token accuracy is 0.475. We show both."
- **"Why Kaggle instead of your own GPUs?"** "Spot-native tranches: pause at an
  exact accepted checkpoint and renew from governed resume state. The recorded
  checkpoint includes optimizer and gradient state; we do not claim that every
  external interruption is lossless. Private kernels, content-addressed packets,
  and full local lineage after fetch make the work inspectable."
- **"Is the memory vector-based RAG?"** "No — it's exact records with
  hash-chained provenance to the source DB row, plus a derived term index.
  Retrieval shown is exact and hash-verified; there is no semantic embedding
  recall serving yet."
- **"What happens on malformed input?"** "It fails closed, transactionally —
  you saw the exception and the unchanged tick. Nothing partial is ever
  written."
- **"Can it talk to me right now?"** "No learned conversation is serving. The
  turn you saw used fixture cores to prove the circulation machinery. That gap
  is exactly what the 512D/1024D milestone funds."
- **"How do you know the substrate vectors are right?"** "The bank is frozen
  and byte-compared against a reference on every run (`v7 conformance: PASS`),
  and the full 1.1M-codepoint Unicode roundtrip passes the local acceptance
  test."

## 6. The ask (Google Cloud)

"We are asking for Google Cloud credits/support for governed D64 diagnostics
and renewable training now, followed by the **512D and 1024D compute
milestones** after D64 closes its current gates. The proposal-workspace
serializer is width-generic and has D64/D128 tests; the canonical field
compiler and learned reader are currently D64, so widening them remains future
engineering rather than a capability claim. What we demonstrated today — exact
substrate transport, transactional governance, provenance memory, and tranche
lineage — is the same discipline we will carry to that scale. We are not asking
you to fund a chatbot; we are asking you to fund measured mechanism."

## 7. What cannot honestly be shown tomorrow (and what replaces it)

| Cannot show | Truthful replacement |
|---|---|
| Live conversation with a learned core | Circulation machinery with fixture cores (Scene 4) + explicit "not serving" statement |
| A learned core committing to canonical state | `test_core_grants_are_never_commit_capable` — cores are structurally barred from writing |
| Live Kaggle job status | Offline catalog (`jobs --json`, local phases clearly labeled) |
| Sub-second full memory integrity verify | Sub-second lookup + hash-verified dereference (0.40–1.30 ms measured); full verify is ~103 s, stated as such |
| `run_axon_heart.py --once` against a scratch root | ATTEMPTED 2026-09-09: fails — a bare scratch root has no dormant evidence index (`DormantGenerationError: generation index does not exist`). Excluded from the demo; the pytest scenes cover the same organs. |
| Any claim that the v1/v2 D64 candidates serve | Paused/lifecycle JSON + adjudication reports; gates failed, non-serving |

## 8. VERIFIED / ATTEMPTED / ASSUMED / DO NOT CLAIM

### VERIFIED (executed by me, 2026-09-09, this machine)

- `python substrate/substrate.py` → `v7 conformance: PASS`, 10 PASS gates.
- Unicode roundtrip `"café 🙂 中文"` → 17 token ids, `roundtrip exact: True`;
  unpaired surrogate → `InvalidUnicodeScalarError`.
- Region-mask heredoc → `0 '' True` / `50 'fghij' True` / `100 'abcdefghij' True`
  (mask never changes canonical body or `field_id`).
- `pytest tests/test_heart_control_plane.py` → **19 passed** (~10 s).
- `pytest tests/test_reasoning_circulation.py` → **5 passed** (~40 s).
- Dormant manifest `record_count` = **59,875**; immutable index meta shows
  427,001 containers / 351,978 edges; episode-container seek + SHA-256 verify
  in **0.40–1.30 ms** across two runs; provenance
  `recovered_corpus_builder:episode_import:v1`,
  source `D:\00\axon_episodic_memory.db:episodes:1`.
- `axon_kaggle.py --state-root "D:/Axon/State" --json jobs` lists both receipt
  jobs with `phase: outputs_fetched` (fully offline).
- v2 lifecycle JSON: `status: paused`, step 120, renewable-tranche reason;
  accepted-step pointer `current_step: 120` with content-hash bundle ids.
- Read-only subprocess recon of trainer state, dormant schema, and launcher
  inventories (documented above with file:line evidence).

### ATTEMPTED

- `run_axon_heart.py --state-root State/tmp/demo-heart-runbook --once --user …`
  → **failed**: bare scratch root lacks a dormant evidence index
  (`DormantGenerationError`). Scene excluded; no state outside `State/tmp`
  was written. Note: this run may have refreshed `__pycache__` mtimes under
  `D:\Axon\runtime` (bytecode cache only, no source change) — flagged for
  transparency.
- `DormantEvidenceIndex.open(r"D:\Axon\State")` (binding-verified bridge open)
  → works but takes ~13.8 s (freshness stat-guard over 3.5 GB sources); too
  slow for stage — the raw immutable-URI query replaces it.

### ASSUMED

- The meeting runs on this machine with this repo and `D:\Axon\State` present
  (mission implies it; §1 covers the fallback if not).
- Google Meet screen share renders UTF-8 terminal output correctly (verify in
  preflight).
- Exact pytest durations will be similar tomorrow (torch import ~30–40 s
  first time).

### DO NOT CLAIM

- Consciousness, understanding, or general intelligence — of anything shown.
- A serving learned Heart or reasoning core; conversational ability.
- A working D128/D512 canonical field compiler or learned reader; only the
  proposal-workspace serializer has verified D64/D128 width-generic coverage.
- Production readiness, deployment, or user-facing product status.
- Learned end-to-end Unicode transport (exact terminated transport is 0.000).
- Sub-second full-corpus integrity verification (lookup only).
- Live cloud status (all Kaggle views are local, fetched evidence).
- Any architecture-tournament result (FFN256 vs others is undecided).
- Any number not in this runbook without re-measuring it first.

---

*Kimi K2.7 Coding / Kimi Code CLI / 2026-09-09 — bounded demo engineer under
Codex, worktree `D:\Axon-worktrees\kimi-google-demo`, branch
`codex/kimi-google-demo-20260909`.*
