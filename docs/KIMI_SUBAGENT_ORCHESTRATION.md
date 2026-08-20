# Kimi Code Sub-Agent Orchestration for Axon

Updated: 2026-08-20

## Purpose

ChatGPT is the supervising project engineer and publication authority for this workflow. Kimi Code is an additional local engineering workforce, not a second project authority. Kimi jobs receive bounded packets, return evidence/diffs/tests, and never commit/push or edit the Axon engineer ledger or `D:\ChatGPT_State`.

## Verified local installation

- executable: `C:\Users\axema\.kimi-code\bin\kimi.exe`
- Kimi Code CLI: `0.36.1`
- default configured model: `kimi-code/k3` (display name `K3`)
- additional configured models: `kimi-code/kimi-for-coding` (`K2.7 Coding`) and `kimi-code/kimi-for-coding-highspeed`
- K3 is the default model used by Axon's supervisor unless an explicit `--model` is supplied.

Do not copy OAuth/token material into Axon. The supervisor relies on the user's existing Kimi Code login/configuration.

## Model routing policy

The supervising Kimi session may use K3 while routine sub-agents use the documented secondary-model pool. The local Kimi config now sets ordinary `kimi-code/kimi-for-coding` (K2.7 Coding) as the default worker model, with K3 and Highspeed available as explicit alternatives. The supervisor enables Kimi's experimental secondary-model feature for its child process.

Usage policy:

- K2.7 Coding is the default workhorse for routine exploration, review, recurring triage, and bounded implementation.
- K3 is preferred for difficult architecture, deep debugging, high-stakes review, and final synthesis.
- K2.7 Highspeed remains available but is opt-in rather than the recurring default.
- If a worker's persisted Kimi wire says it actually bound a different model, report the observed model rather than the intended policy.

The first 7-lens AgentSwarm on 2026-08-20 exposed why this explicit pool matters: the attempted environment-only secondary-model routing did not take effect, all started workers bound to K3, and Kimi exhausted the billing-cycle quota after four of seven lenses completed. That run is preserved under `State/kimi_orchestrator/jobs/20260820t2102z-full-repo-swarm/`.

## Project-scoped agents

Kimi automatically discovers project agents below `.agents/agents/`.

- `axon-explorer` — read-only code/state/doctrine mapping.
- `axon-architect` — read-only architecture planning; may delegate to explorer through Kimi's `Agent` tool.
- `axon-triage-lead` — read-only recurring K3 coordinator; delegates isolated explorer/reviewer/architect views and merges them into one next packet.
- `axon-reviewer` — read-only independent invariant/drift review; may delegate narrow exploration.
- `axon-coder` — bounded implementation; may edit/run tests, but never commit/push, launch training, promote checkpoints, or edit continuity ledgers.
- `axon-swarm-lead` — K3 coordinator for multi-context implementation work; delegates to bounded agents and requires independent review after implementation.

These profiles supplement `AGENTS.md`; they do not override Jeffrey or `docs/SOURCE_OF_TRUTH.md`.

## ChatGPT supervisor

Use:

```powershell
python scripts\run_kimi_roundtable.py --mode audit --mission "<single read-only audit>"
python scripts\run_kimi_roundtable.py --mode triage --mission "<multi-agent read-only triage>"
python scripts\run_kimi_roundtable.py --mode plan --mission "<architecture question>"
python scripts\run_kimi_roundtable.py --mode review --mission "<review packet>"
python scripts\run_kimi_roundtable.py --mode implement --mission "<bounded implementation packet>"
```

A packet file inside the repository may be supplied with `--mission-file` instead of `--mission`.

Default model is K3. Override only deliberately:

```powershell
python scripts\run_kimi_roundtable.py --mode review --model kimi-code/kimi-for-coding --mission "<packet>"
```

The supervisor refuses a dirty worktree by default and maintains an exclusive `State\kimi_orchestrator\active.lock.json` while Kimi is running. Durable job records live under `State\kimi_orchestrator\jobs\<job-id>\` and include prompt, stdout, stderr, status, hashes, Git HEAD before/after, and final Git status. State is ignored by Git.

`--allow-dirty` exists only for a supervising ChatGPT session that has already inventoried the dirty worktree and is deliberately running a read-only job against it. Do not use it for unattended implementation.

The historical `scripts\run_supervised_kimi_packet.py` command is retained as a compatibility wrapper and now delegates to this supervisor rather than embedding stale July continuity or hard-coded user paths.

## Why `kimi -p` works for supervision

Kimi's non-interactive prompt mode uses its automatic permission policy. Axon's project agent profiles therefore provide tool restrictions and mission boundaries; the outer supervisor provides dirty-tree/concurrency guards and captures durable evidence. ChatGPT still independently reviews changed files and tests before any commit or push.

Kimi sub-agents run with isolated context windows. The parent must pass each sub-agent a complete bounded question and receives only the returned result, which is useful for independent exploration/review without contaminating the lead context.

## Recurring contribution policy

Recurring work should be **read-only by default**. It is safe and useful to automate repository/architecture drift audits, blocker discovery, stale-path searches, test-failure triage from existing output, and creation of a recommended next packet.

Do not schedule unattended Kimi code-writing, long test/training jobs, service changes, checkpoint promotion, or automatic Git publication. Those actions should remain explicit supervised missions because Axon is changing rapidly and the Working Contract requires freshness/concurrency checks.

Recommended recurring cycle:

1. Re-read `D:\ChatGPT_State` and live Axon authority/ledger.
2. Verify no other mission owns the machine and the Git worktree is clean.
3. Run one supervised `triage` job through `scripts\run_kimi_roundtable.py`; ordinary K2.7 Coding is the default sub-agent workhorse while the lead may remain K3 for synthesis.
4. ChatGPT reviews Kimi's durable result and selects one bounded next packet.
5. No code is changed automatically by the recurring triage.
6. When Jeffrey or ChatGPT explicitly authorizes the packet, run `plan` / `implement` / `review` as one supervised mission.
7. ChatGPT owns final tests, ledger update, commit, push, and carried-state refresh.

This gives continuous useful pressure toward completion without allowing an unattended agent to drift architecture or collide with another engineer.
