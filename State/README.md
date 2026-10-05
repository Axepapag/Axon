# State/

Axon's single living State root for this repository. Heart-governed branches (Shared Field, Souls, Dormant memory,
training runs) are created beneath it. Everything here is machine-local and git-ignored except this file.

Rules (inherited from the old repo, unchanged):

- There is exactly one canonical state body per branch. No component may create a competing Axon state universe.
- Dormant memory (exact records) is the authority for what happened; any index over it is disposable and rebuildable.
- Derived views never become truth; the Heart is the sole writer.
- Anything archived here is evidence, never live authority, and is never resumed or promoted implicitly.

Defaults: `runtime/field/state_branch.py` (`DEFAULT_STATE_ROOT`) and `runtime/soul/store.py` (`DEFAULT_SOUL_ROOT`)
point here. Tests use temporary folders and never touch this one.