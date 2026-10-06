# E0 catalog and Core readiness delivery

Codex integrated Core-owned E0 v0.1.0 manifests into the default API catalog.
The public capabilities endpoint exposes all three contracts. Edge validation
requires trusted surface declarations and rejects substrate-exact/free crossings
unless the graph contains an explicit registered versioned adapter. The API
contract records the agreed surface amendment and WAIT/control distinction.

Startup evidence runs in a separate CPU process with a 90-second deadline,
disposable architecture registry, fixed seed and no optimizer. Cached results
carry a hash of inspected source. Checks prove adapter import, D512 graph API
validation/registration/readback, and identical checkpoint weights, both states,
configuration and next-tick outputs. Foundation preflight is unchanged. Training
authorization remains false; runtime/run/inference/checkpoint endpoint families
still refuse unavailable capabilities. No valuable training ran.

Verification: full pytest suite 188 tests passed; both frozen substrate self-tests
exited 0. Focused tests cover real default catalog/readiness/registration, private
smoke storage, training lock, surface mismatch, missing surface and explicit
versioned-adapter wiring. Backend/contract/test whitespace check passed. Global
git diff --check reports pre-existing immutable canonical-ledger whitespace;
historical records were preserved.

Deployment: managed main-sites child PID29636 was verified under supervisor10700
before replacement. New child PID27552 serves the public UI/API on8080. Tunnel
PID21916 and all other managed service PIDs remained unchanged. Public health,
capabilities and readiness were verified. All three Core checks returned passed,
scoped to CPU D512, with training_authorized false. Startup briefly refused a
request while the bounded smoke process completed; subsequent public API checks
succeeded.

Owner follow-ups sent to KimiCode: reference graph version integer requires
explicit text normalization at API registration; variable configurations are
advertised while ports remain fixed D512/64; occupancy limits need alignment.
Readiness verifies only D512/hidden512/occupancy1. The checkpoint loader's fallback
to unrestricted deserialization was flagged: adjacent SHA256 proves integrity,
not trusted provenance. Core-owned source was not modified by Codex.

ChatGPT acknowledged frontend ownership through Jeff's Agent Browser Hub and
is working on lost-ack preflight recovery, stale-poll/auxiliary-refresh behavior,
README correction and real-manifest builder checks. Perplexity was asked for
current-source/untracked-work and ignored-artifact backup evidence. Neither
frontend fix completion nor independent backup completion is claimed here.
