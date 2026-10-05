# Axon Engineer's Ledger Protocol

Established: 2026-08-17
Authority: Jeff, project convener
Applies to: every agent and human doing work within the Axon repository or on
the Axon project.

## Purpose

Axon maintains two complementary ledgers:

1. `ENGINEERS_LEDGER.md` is the rolling continuity summary. It is deliberately
   rewritten as the project changes so a new participant can quickly recover
   the current state, active decisions, risks, and next work.
2. `ENGINEERS_LEDGER_CANONICAL.jsonl` is the never-deleted, never-altered,
   append-only historical record. Every completed, failed, interrupted, or
   read-only project turn appends exactly one event.

The rolling summary is a derived convenience. The canonical JSONL ledger is
the historical authority. If they disagree, the canonical events win and the
rolling summary must be corrected.

## Required workflow for every turn

### At turn start

1. Read this protocol.
2. Read the rolling summary.
3. Read enough of the canonical ledger tail to identify work after the
   rolling summary's `current_through_event_id`.
4. Reconcile stale context before making changes.

### During the turn

Keep a list of every material action. A material action includes:

- reading, searching, or inspecting project files or state;
- running a command, test, evaluation, training job, migration, or service;
- creating, editing, moving, archiving, or deleting a file;
- changing configuration, dependencies, Git state, runtime state, datasets,
  checkpoints, or remote infrastructure;
- making a decision, discovering evidence, raising a flag, or changing the
  intended next step;
- sending project instructions or results to another participant.

Incidental mechanics such as polling an already-recorded command do not need
separate action items, but their final result does.

### Before ending the turn

1. Append one single-line JSON event to the canonical ledger. Never modify an
   earlier line.
2. Include all material actions and their outcomes in that event.
3. Rewrite the rolling summary to incorporate the new verified state.
4. Set `current_through_event_id` in the rolling summary to the appended event.
5. Report both ledger updates in the participant's normal final response.

If a process crashes before the turn event can be appended, the next
participant records a recovery event describing the orphaned evidence.

## Canonical event schema

Each physical line is one complete UTF-8 JSON object with these required
fields:

```json
{"schema":"axon-engineers-ledger-event-v1","event_id":"evt-YYYYMMDDTHHMMSSffffffZ-agent-slug","timestamp":"ISO-8601 timestamp with offset","agent":{"name":"participant name","model":"model or human","session":"session/thread identifier or unknown"},"turn":{"request":"concise request summary","status":"completed|partial|blocked|failed|interrupted","summary":"concise outcome"},"actions":[{"kind":"read|search|command|test|create|modify|move|archive|delete|decision|communication|other","target":"path, system, or subject","summary":"what happened","result":"verified outcome"}],"files":{"created":[],"modified":[],"moved":[],"deleted":[]},"verification":["evidence produced this turn"],"decisions":["decisions made or confirmed"],"flags":[],"next_steps":[],"identity_stamp":"name / model / date"}
```

Additional fields are allowed. Required arrays remain present even when
empty. Paths should be repository-relative when possible. `event_id` values
must be globally unique; UTC time with microseconds plus an agent slug is the
preferred form.

Do not store secrets, credentials, personal data that does not belong in the
repository, or huge command output in the ledger. Record a hash or durable
artifact path instead.

## Immutability and corrections

- Existing canonical lines are immutable, including spelling and formatting.
- Never truncate, rotate, squash, reformat, sort, or deduplicate the canonical
  file.
- Never use a history rewrite to remove a canonical event.
- A mistake is corrected by appending a new event with
  `corrects_event_id` and an explanation.
- A reversion is a new event; it does not erase the original action.
- If concurrent branches append events, merge by retaining every complete
  line. Record the merge as another event if it changes project history.
- If sensitive material is accidentally recorded, halt and ask Jeff to govern
  the exceptional remediation. Do not silently alter history.

## Rolling summary rules

The rolling summary may be freely rewritten, but it must stay compact and
evidence-based. It should contain:

- the canonical event through which it is current;
- current mission and project state;
- binding decisions and invariants;
- recent completed work;
- active work, blockers, and risks;
- verified tests and evaluations;
- important paths and commands;
- the next recommended actions.

Do not turn the rolling summary into a second chronological archive. Details
belong in the canonical ledger.

## Relationship to existing contracts

This protocol supplements `docs/WORKING_CONTRACT.md` and does not change Axon
runtime doctrine. The existing requirements for truth in reporting,
provenance, flags, protected ground, and identity stamps still apply.
