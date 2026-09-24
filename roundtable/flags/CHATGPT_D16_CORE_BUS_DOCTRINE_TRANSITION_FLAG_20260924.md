# FLAG [BLOCKING] — D16 Core Bus doctrine transition

Identity: ChatGPT / GPT-5.6 Sol / 2026-09-24 America/Chicago

Mission item: Move Axon reasoning Cores from width-specific packed reasoning rails to a Heart-served exact D16 delta bus with persistent Core-local exact Shared Field mirrors.

Problem: doctrine conflict during implementation transition.

Evidence:
- Jeff explicitly directed on 2026-09-24 that reasoning Cores should speak directly in 16 dimensions, receive Heart-served deltas, maintain synchronized local Shared Field mirrors, and use a Heart-served proposal collection bus rather than the D64 reasoning rail.
- `docs/SOURCE_OF_TRUTH.md` currently marks the packed-rail pivot as binding doctrine. Lines 300-336 require exact 16D cells to be packed into width-specific rails and state that Cores attend their designated rail.
- `docs/SOURCE_OF_TRUTH.md` lines 352-357 currently require exact English proposals to be mechanically repacked by Heart onto destination rail widths.
- `runtime/heart/circulation.py` currently binds each `ReasoningPassRequest` to a `RailRuntimeView` whose width must equal the Core descriptor `d_model`.
- `runtime/heart/proposal_workspace.py` currently stores an exact noncanonical English proposal board but renders it into width-specific `RenderedProposalRail` objects.

Options I see:
1. Codify Jeff's new D16-bus ruling through the RoundTable, amend the conflicting Source-of-Truth sections, then change the serving runtime while preserving the D64 implementation as non-serving legacy evidence.
2. Add a parallel experimental D16 bus behind a non-serving feature seam while leaving the packed-rail serving path authoritative until doctrine is amended.
3. Keep the new design proposal-only and make no runtime changes.

My recommendation: Option 1. The isolated `D:\ContinuousCoreLab` evidence strongly supports a direct exact-D16 input/output boundary with a wider private recurrent chamber, and the existing Axon circulation/commit machinery can largely be preserved if only the Core transport seam is replaced.

Work halted: Runtime code changes that would bypass or remove the current packed-rail serving contract; Source-of-Truth edits not yet ratified through the table.

Work continued: Verified the isolated lab results, inspected current Axon governance and runtime seams, mapped reusable versus rail-bound components, and drafted the companion D16 bus/mirror-coherence ratification candidate.

Important limitation: `D:\ContinuousCoreLab` used an isolated synthetic frozen 16D codebook, not Axon's canonical substrate. No lab checkpoint, model state, or lab source file has been imported into Axon.
