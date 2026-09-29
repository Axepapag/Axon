# Proposal — Ledger consolidation, compression, and summarization

**Author:** Claude / Claude Code
**Date:** 2026-09-27
**Status:** PROPOSED — awaiting Jeff's ruling. Nothing in this document has been executed.
**Trigger:** Jeff, 2026-09-27 — *"the ledgers are growing pretty significantly. So there's going to have to be a way of consolidating and compressing and summarizing, while leaving the canonical ledgers immutable for our records."*

---

## 0. Summary

The problem is **not** that the ledgers are large. It is that two of the three have
stopped being what they are defined to be:

- `ENGINEERS_LEDGER.md` is defined as *"a derived convenience"* that *"must stay compact"*
  and must **not** become *"a second chronological archive"*
  (`roundtable/ENGINEERS_LEDGER_PROTOCOL.md:19-21, 95-108`). It is 307,844 bytes, 53.3% of
  which is dated chronological narrative. **It is now the thing the protocol forbids.**
- `ENGINE_TEAM_BUS.md` is 96.6% pre-2026-09-15 and has been dormant for 7 days.

The canonical ledger is **not** the problem and must not be touched. It is append-only by
three binding documents plus a `.gitattributes` byte-preservation rule, and it has in fact
never lost an event. Its growth is healthy; its *reader* is not.

**The asymmetry is the strategy.** The rolling summary is contractually free to rewrite;
the canonical ledger is contractually frozen. So the work is: compact what may be compacted,
and build a *derived* reader for what may not.

**The governing principle is Axon's own.** Source of Truth already defines the correct
pattern for exactly this situation — an exact immutable scaffold plus a *derived,
rebuildable* semantic layer that *"never become[s] the only copy of anything"* and *"hold[s]
no reasoning vote and no commit authority"* (`docs/SOURCE_OF_TRUTH.md:1764-1772`). The ledger
should obey the same law as the field it records.

---

## 1. Measured state (VERIFIED)

All figures measured this session; commands reproducible.

| Artifact | Size | Count | Status |
|---|---|---|---|
| `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` | 2,380,970 B | **410 events** (411 lines; last is empty) | Healthy. Immutable. Leave alone. |
| `roundtable/ENGINEERS_LEDGER.md` | **307,844 B** | 4,043 lines, 54 top-level headings | **Violates its own charter.** |
| `roundtable/ENGINE_TEAM_BUS.md` | 130,008 B | 786 lines | **96.6% stale, dormant 7 days.** |

### 1.1 The rolling summary broke on one specific turn

The summary oscillated between 1,928 B and 35,589 B (median ≈ 16 KB) from 2026-08-17 to
2026-09-14 — it **behaved as designed**, shrinking as often as it grew. Eight same-day
compactions are visible in git (e.g. 35,589 → 8,670 on 09-13; 6,017 on 09-14).

Then it became strictly monotonic and has **never decreased since**:

| Date | Bytes |
|---|---|
| 2026-09-14 | 6,017 |
| 2026-09-17 | **170,512** ← regime change |
| 2026-09-18 | 216,882 |
| 2026-09-24 | 273,689 |
| today | **307,844** |

The break is a **single commit**: `1a4bc41` (2026-09-17, *"Teach Stage-0 emission and attribute
every training surface"*) added **+1,456 / −90 lines to `ENGINEERS_LEDGER.md` alone** — roughly
145 KB of narrative written into a file whose contract says *"details belong in the canonical
ledger."* Every subsequent turn appended to it instead of rewriting it. **28× growth in 9 days.**

This matters for the remedy: the failure is a **lost rewrite discipline**, not a slow leak. It
is correctable by resuming the practice, not by inventing new machinery.

### 1.2 What actually occupies the space

For the canonical file, the mass is **prose, not schema**:

| Component | Bytes | Share |
|---|---|---|
| `action.result` + `action.summary` | 602,543 | **25.3%** |
| `turn.summary` | 387,766 | **16.3%** |
| all near-empty/rare fields combined | 29,591 | **1.2%** |

Mean 5,806 B/event; median 5,264 B; max 17,249 B. Growth is ~**50–65 KB/day**, and the mean
event size is **flat** month over month (5,886 B in Aug vs 5,761 B in Sep).

> **Consequence: schema dieting is a dead end.** Deleting every rarely-used field buys 1.2%.
> Any real reduction must be a change in *representation*, not in schema.

### 1.3 The growth is already a functional problem, not an aesthetic one

`docs/AXON_HOME_ARCHITECTURE.md:455` and its risk table at `:531` already record that the
canonical JSONL *"~2.1 MB exceeds the mission's push transport"*, that there is **no remote
ledger-ingestion endpoint**, and that concurrent appends are hazardous. The file is now
**2.38 MB — approximately 280 KB past the documented ceiling**, growing ~58 KB/day. This
constraint is internal to the repo; it is not an outside observation.

### 1.4 What is genuinely healthy and should not be "fixed"

- **The canonical ledger has never lost or reordered an event.** Verified independently by
  reconstructing all 235 revisions and confirming each is a strict prefix-extension of its
  predecessor: **0 non-append commits.**
- `current_through_event_id` is **currently accurate** (`evt-20260927T025500000000Z-chatgpt-web-monitor`
  is genuinely the last line).
- `ENGINE_TEAM_BUS.md` has **added=786, deleted=0** across 43 commits. The convention has been
  honoured without exception.

### 1.5 Defects found while measuring (see §5)

1. **`append_engineers_ledger_event.py:45` re-parses all 410 lines on every append** — an O(n),
   quadratic-over-time cost on a file that grows forever.
2. **Two whole-file line-ending flips** (`64c0b1a` 2026-09-06 LF→CRLF; `aff18b6` 2026-09-19
   CRLF→LF). The second happened **12 days after** `.gitattributes` asserted `-text` on the file,
   so an editor or tool bypassed the rule. Both produced whole-file phantom diffs. No event was
   lost, but the guard failed.
3. **`kind` is not a closed vocabulary** — 41 distinct values, **76 non-dict action items**,
   14 with `kind: null`. The schema in the protocol lists nine kinds.
4. **`event_id` format is inconsistent** with the protocol's own preferred form (line 72-73) —
   some ids lack microseconds, some carry spurious trailing zeros (`T012800000000Z`), and ids
   claim `Z` (UTC) while `timestamp` carries `-05:00` (local).
5. **Byte-level encoding corruption** in `ENGINEERS_LEDGER.md` headings — em-dashes render as
   `?` (e.g. line 1, `## 2026-09-17 ? the false-progress trap`).

---

## 2. The asymmetry that defines the strategy

Three binding documents were read directly. They draw a sharp line:

**The canonical ledger is frozen.**
- `ENGINEERS_LEDGER_PROTOCOL.md:80-84` — *"Existing canonical lines are immutable… Never truncate,
  rotate, squash, reformat, sort, or deduplicate the canonical file. Never use a history rewrite
  to remove a canonical event."*
- `docs/WORKING_CONTRACT.md:170-171` — *"Never alter, delete, reorder, compact, or redact an
  existing canonical event."*
- `AGENTS.md:21-25` — conflicts *"must be resolved by preserving every valid event, never by
  choosing one side or rewriting existing events."*
- `.gitattributes` — `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl -text`.

**The rolling summary is explicitly free.**
- `ENGINEERS_LEDGER_PROTOCOL.md:19-21` — *"The rolling summary is a derived convenience. The
  canonical JSONL ledger is the historical authority. If they disagree, the canonical events win
  and the rolling summary must be corrected."*
- `:54` — the required workflow is *"Rewrite the rolling summary to incorporate the new verified
  state."*
- `:95-108` — *"The rolling summary may be freely rewritten, but it must stay compact and
  evidence-based… Do not turn the rolling summary into a second chronological archive."*

**Therefore: compacting the rolling summary requires no doctrine change and no ruling. It is a
compliance action, currently overdue by a factor of ~15×.** Consolidating the canonical ledger
*is* forbidden without an exception Jeff must grant. This proposal does not request one.

---

## 3. The strategy — three tiers, one law

**Law: the canonical ledger is the exact scaffold; every consolidation is a derived, rebuildable
view over it that cites exact `event_id`s and holds no authority.** This is `SOURCE_OF_TRUTH.md:1764-1772`
applied to the ledger.

| Tier | Artifact | Mutability | Authority |
|---|---|---|---|
| **T0 — exact** | `ENGINEERS_LEDGER_CANONICAL.jsonl` | **Never written except by append** | **Sole historical authority** |
| **T1 — derived index** | `roundtable/ledger_index/*` (new) | Regenerated at will, always from T0 | **None.** Cites T0. Deleteable without loss |
| **T2 — rolling summary** | `ENGINEERS_LEDGER.md` | Freely rewritten every turn | Convenience only; corrected by T0 |

### T0 — exact (unchanged, and protected harder)

No content change, ever. Two additive protections are proposed:

- **A CI guard** that fails if the canonical file's line count decreases, or if the first N−1
  lines are not byte-identical to `HEAD~1`. This is cheap, and it would have caught both CRLF
  flips immediately. It enforces an invariant the repo already claims but does not check.
- **Stop the O(n) re-parse** at `append_engineers_ledger_event.py:45`. Uniqueness against the
  whole file is the right *semantics* but the wrong *implementation*. A tail-scan (last K lines)
  plus a periodic full audit gives the same guarantee at O(1) amortized. **This is a code change,
  not a doctrine change.**

### T1 — derived index (new; requires ruling to adopt)

A machine-generated directory, rebuilt from T0 by one script, never hand-edited:

- `by_era.json` — events grouped into the project's own eras (Heart → shelf pivot → receipt →
  English-native → D16 Core Bus), each with a 200-word digest and its exact event-id range.
- `by_topic.json` — index of `event_id` → topic tags, so a question like "when did attention get
  ruled out of B4" returns ids, not prose.
- `open_threads.json` — every `next_steps` and `flags` entry that has no subsequent resolving
  event. This is the highest-value view: it replaces the current 25 KB `## Next actions` archive
  with a query.

**Rules that make it doctrine-safe:** it is regenerable from T0 alone; every claim carries
`event_id`s; it is explicitly marked non-authoritative in its own header; and deleting the whole
directory loses nothing.

### T2 — rolling summary (compact now; no ruling needed)

Rewrite `ENGINEERS_LEDGER.md` from 307,844 B to a **hard budget of 20 KB**, with a fixed
skeleton and nothing else:

1. `current_through_event_id` (+ updated stamp)
2. Current mission and project state — ≤ 4 KB
3. Active blockers and risks — ≤ 3 KB
4. Settled doctrine changes since the last compaction, each with its `event_id` — ≤ 4 KB
5. Open questions awaiting Jeff — ≤ 3 KB
6. Next actions — ≤ 3 KB
7. Pointer to T1 for history

**Method (important):** compaction is *not* deletion. Narrative currently in T2 is not lost —
it is either (a) already in T0 as the event it came from, or (b) promoted into a T1 digest with
its event ids, or (c) if genuinely only in T2, written back as a **new canonical event** first.
That ordering matters: **capture before compact.**

Proposed enforcement: a per-turn check that the file is under budget, surfaced in the same place
the ledger append reports. If it is over budget, the turn is not complete.

---

## 4. Sequenced plan

| # | Action | Ruling needed? | Risk |
|---|---|---|---|
| 1 | Fix root `SOURCE_OF_TRUTH.md` mirror divergence (§5.1) | **Yes — doctrine is protected ground** | — |
| 2 | Capture any T2-only narrative as canonical events | No | Low |
| 3 | Compact `ENGINEERS_LEDGER.md` → ≤ 20 KB | **No** (protocol §95-108 authorizes) | Medium — irreversible in the sense that prose leaves the file, hence step 2 first |
| 4 | Archive `ENGINE_TEAM_BUS.md` pre-2026-09-15 sections to `archive/` | No (bus is non-canonical by its own text; contract says *archive, never delete*) | Low |
| 5 | Add the canonical-ledger CI guard | No | Low |
| 6 | Fix the O(n) append re-parse | No | Low |
| 7 | Adopt T1 derived index | **Yes** | Low |
| 8 | Add per-event field budgets to the protocol | **Yes — protocol change** | — |

Steps 3, 4, 5, 6 are executable today. Steps 1, 7, 8 are table matters.

**Note on step 4:** `ENGINEERS_LEDGER_PROTOCOL.md:80-84` forbids touching *canonical* lines. The
bus is not canonical — it states at lines 5-8 that it is *"for coordination… not canonical
history"* — so archiving it is not a protocol exception. Its bytes must be preserved verbatim in
the archive file; nothing is deleted.

---

## 5. Live governance defects found during this sweep

These are **not** part of the consolidation strategy. They are flagged because a
consolidation proposal is worthless if the authorities it consolidates are already
inconsistent.

### 5.1 Root `SOURCE_OF_TRUTH.md` is a stale second authority — **HALT AND FLAG**

Doctrine requires the two mirrors to be **byte-identical**: *"Any doctrine update must update both
in the same change; repository tests enforce equality"* (`docs/SOURCE_OF_TRUTH.md:2032`).

They are not. `tests/test_day_zero_hygiene.py::test_authority_mirrors_are_byte_identical`
**FAILS** (VERIFIED — executed this session):

```
assert (ROOT / "SOURCE_OF_TRUTH.md").read_bytes() == (ROOT / "docs" / "SOURCE_OF_TRUTH.md").read_bytes()
AssertionError: At index 46 diff: b'1' != b'2'
```

- root `SOURCE_OF_TRUTH.md` — 150,858 B, last touched **2026-09-19** (`8825dd3`)
- `docs/SOURCE_OF_TRUTH.md` — 157,485 B, last touched **2026-09-24** (`b8bb438`)

**The mirror discipline held until 2026-09-24 and broke that day.** Commit `8825dd3` (09-19)
correctly updated both. Commits `f0e3d9b` (**+97 lines — the D16 Core Bus ratification**) and
`b8bb438` (+21 lines) updated **`docs/` only**:

```
f0e3d9b 2026-09-24 Axon: ratify and prove D16 Core bus
    docs/SOURCE_OF_TRUTH.md   | 97 ++++-
b8bb438 2026-09-24 Axon: add fresh D512 continuous Core baseline
    docs/SOURCE_OF_TRUTH.md   | 21 +-
```

**Consequence:** the root mirror is missing the entire ratified *"D16 Core Bus and resident mirror
coherence (2026-09-24)"* section and still carries superseded packed-rail wording. An agent
following `AGENTS.md`'s instruction to read the architecture will read correct doctrine; an agent
that greps the repo root will read **superseded doctrine**. The test that exists to catch this has
been red since 2026-09-24. It is not the only day-zero guard currently failing — see §5.5 — but it is
the only one whose cause is a genuine doctrine divergence.

**Doctrine is protected ground (`WORKING_CONTRACT.md:66-68`). This is recorded, not fixed.**

### 5.2 "Layer 13" is a dangling reference

`docs/WORKING_CONTRACT.md:96,100` cites *"Training doctrine floor (locked, from SOURCE_OF_TRUTH
Layer 13)"*, and `docs/SOURCE_OF_TRUTH.md:516,558` cite Layer 13 as binding.

**No numbered layer structure exists in SOURCE_OF_TRUTH** — its headings are topical, and
historical revisions were topical too. The floor's *content* survives only as a quotation at
`docs/WORKING_CONTRACT.md:102-103`. The table has been citing a section number whose definition
no longer exists in the document it names. Low urgency, but it should be repaired by giving the
floor a real heading.

### 5.3 The README's headline receipts advertise a retired mechanism

`README.md:30-40` presents **"Verifiable Cloud Receipts (D64 Receipt Continuation)"** as a
front-page proof, with statuses **SOLVED / SOLVED / RECOVERED / −81.3%**. The header names
*D64 Receipt Continuation* explicitly.

That mechanism was **removed from doctrine on 2026-09-19** (commit `e09578c`, *"THE LEARNED
DELTA / NO_OP / ABSTAIN DECISION HEAD IS REMOVED FROM AXON DOCTRINE"*) and its code archived to
`archive/legacy_typed_reasoning_20260919/`. Source of Truth now states that the whole
typed-reasoning line *"must not receive further reasoning-training budget, be promoted, or be used
as the target interface for new learned cores"* (`docs/SOURCE_OF_TRUTH.md:874-879`).

The table carries **no retirement notice**, while its own citation line
(*"immutably recorded in `roundtable/reports/`"*) points to a directory whose newest contents
postdate the retirement. This is a public-facing document on a GitHub repository presenting a
retired mechanism as a solved capability — the same class of overclaim the working contract's
*"proof over proxy"* rule (`.100-110`) exists to prevent.

Related, lower confidence: the README badge claims **"594 passing"** tests. The repo now contains
**720 `def test_*` functions across 83 files** (counted, not executed), and at least one test fails
(§5.1). So the badge is stale at minimum, and the suite is not fully green. **ASSUMED** — the full
suite was deliberately not run, because a training run may be live and the contract requires one
mission per machine at a time.

### 5.4 The canonical ledger's tail is uncommitted, and four completed runs are unrecorded

Found while auditing training state. Two separate issues, and they should not be conflated:

**(a) The last 11 events are uncommitted.** `git status` shows
`roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` modified with **+11 / −0 lines** — eleven genuine
events covering 2026-09-25 to 2026-09-26 (GRU breath training, language EOS audit, v3, v4,
persistent generation, stream-school launch, web monitor). The content is present and
protocol-shaped; it simply has not been committed. `ENGINEERS_LEDGER.md` is likewise modified.

One of those eleven carries a malformed timestamp — `evt-20260926T153327907945Z-chatgpt-stream-school-launch`
has `"2026-09-26T10:33:27.8979462-05:00"`, i.e. **seven fractional digits**, which is not valid
ISO-8601. Its `event_id` also encodes `…907945Z` against a timestamp of `…8979462`, a 10 ms
disagreement. Minor, but it is exactly the class of drift a schema check would catch — see §3's
proposal to add one.

**(b) Four completed runs have artifacts but no event.** `grep schoolv6` over the ledger returns
**0 hits**, and the two most recent checkpoint ids (`d8812bb7`, `8e5756b3`) appear nowhere:

| Run | Artifact | Event |
|---|---|---|
| v5 stream-language **completion** | yes | the launch event is marked *"partial … pilot that is running"*; completion never recorded |
| mixed-language v6 smoke | yes | none |
| school v6 smoke | yes | none |
| school v6 400-step / 128,000-target tranche | yes (`d8812bb7`) | none |

The newest event is the **monitor**
(`evt-20260927T025500000000Z-chatgpt-web-monitor`), timestamped *after* training stopped — so the
ledger's tail describes the dashboard rather than the work.

This conflicts with `docs/WORKING_CONTRACT.md` §12 (*"record every material action in one new
canonical turn event"*), and it is directly material to this proposal: **a derived index can only
be as complete as the exact tier beneath it.** Consolidation cannot recover evidence that was never
recorded. Ordering therefore matters — commit the pending eleven, then record the four missing
runs, *then* compact:

### 5.5 Two further day-zero hygiene guards are red, and the tree was left non-green

The newest file in the tree is a failing test sweep:
`logs/pytest_hermes_full_sweep2_20260926.log`, mtime **2026-09-26 22:00:37** — later than the newest
canonical event (21:55). It ends `EXIT=1` with **three failures, all in
`tests/test_day_zero_hygiene.py`**. The same suite that afternoon (14:58) had **twelve** failures, so
the evening cut twelve to three — but the survivors are exactly the guards that police how wide the
active surface is:

| Guard | Failure | Reading |
|---|---|---|
| `test_authority_mirrors_are_byte_identical` | byte 46: `b'1' != b'2'` | real defect (§5.1) |
| `test_day_zero_active_python_surface_is_narrow` | extra file `runtime/field/d16_view.py` | **stale guard** — the D16 view is ratified 2026-09-24 work the whitelist never learned |
| `test_parallel_pre_day_zero_bodies_are_not_live` | `ops` exists and is tracked (`ops/windows_vm/bootstrap_axon_vm.ps1`) | real violation of the pre-Day-Zero-body ban |

Three different problems, three different fixes, and bundling them would be a mistake: one needs a
mirror repair, one a whitelist amendment through the table, and one a decision about whether `ops/`
is a live body or an archive. Until they are separated, *"the day-zero hygiene tests are red"* is not
actionable.

This shares a cause with §5.4. The last ~1.5 h of 09-26 was not training: five `State/tmp/probe_*.py`
scripts landed between 21:14 and 21:58 (`probe_extract_extra`, `probe_recovery_report`,
`probe_recovery2`, `mail_check`, `probe_school6_grade`) plus `sot_mirror.diff`. All of it was
ad-hoc verification of the just-finished run, none of it was captured, and the run was left paused
with that verification unfinished.

---

## 6. What this proposal explicitly does NOT do

- It does not touch a single canonical line, and it requests no exception to do so.
- It does not delete anything. The bus is *archived*; narrative is *promoted* to canonical events
  or to T1 digests before T2 is compacted.
- It does not propose a cap on the canonical ledger's size, per the no-tissue-ceilings law. The
  ledger may grow forever; the point is that its *readers* must not.
- It does not amend Source of Truth.

---

## 7. The question for Jeff

Three rulings are requested, and one is urgent:

1. **Approve this proposal's shape** — T0 exact / T1 derived / T2 compact, with the canonical
   ledger untouched.
2. **Authorize the rolling-summary compaction** (step 3) and the bus archive (step 4), or defer.
   Both are already permitted by the protocol; asking because they discard 287 KB of accumulated
   narrative from a live file.
3. **Rule on the `SOURCE_OF_TRUTH.md` mirror** (§5.1). The safe repair is to copy `docs/` over the
   root mirror, restoring byte-identity and greening the test — but that changes a protected
   surface, so it is yours to call. It is the one defect here that is actively misleading readers
   today.

*Nothing in this document has been executed. This turn was read-only apart from the creation of
this proposal file.*
