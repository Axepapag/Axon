# Proposal: RNSC Attended Soul — Private Persistent Memory Tokens Across Ticks

Author: Perplexity / 2026-09-23 America/Chicago
Status: RoundTable proposal for review; no optimizer or architecture migration authorized
Parent context: RNSC dual-lane rebuild discussion plus current Soul implementation and delayed-recall question

## Thesis

The Soul should function as a private, persistent, temperature-layered episodic memory that is updated during reasoning and causally conditioned on in subsequent ticks. It must not enter through the public Shared Field, but it should enter the Core's reasoning chamber as private tokens so learned attention can operate over external field evidence and internal memory simultaneously.

The current Soul implementation has three practical weaknesses:

1. The temperature layers are projected and combined into one recurrent state rather than remaining separately addressable memory.
2. Existing ablation evidence shows an active but weak causal path: removing Soul changes the distribution slightly without changing the failed pointer verdict.
3. Early single-turn curricula usually contain every fact needed in the current field, so task success does not require using Soul.

## Proposed inhale/exhale behavior

At a causal boundary, the Core inhales its private Soul directly from its private store, not through Shared Field or a public rail. The Soul should be represented inside the reasoning chamber as reserved memory tokens rather than being collapsed into one additive state.

At the end of a reasoning phase or tick, the Core should encode accumulated reasoning state into structured private memory slots. The proposal suggests preserving the four permanent temperature layers:

- HOT: immediate preceding thought trace, unresolved goals, and recent working memory; updated each tick or causal pass.
- WARM: medium-lived memory updated over several ticks by a gated or residual mechanism.
- COLD: longer-lived consolidated episodic material.
- DEEP_COLD: slowly changing distilled material suitable for later consolidation or parameter distillation.

The precise slot counts and update rules are architecture parameters rather than public Shared Field coordinates.

## Causal-use requirements

Architecture alone does not prove that a Core uses its Soul. The curriculum and evaluator must contain tasks for which the needed evidence is absent from the current Shared Field.

Required probes should include:

- Multi-tick delayed recall: present a clue or constraint at an earlier tick, remove it from later Shared Field input, and require the later answer to depend on retained private memory.
- Counterfactual Soul swaps: hold the current field fixed while swapping one Core's Soul for another compatible Soul; output should change in the direction implied by the substituted prior experience.
- Bottlenecked streaming recall: present information in an earlier field window and ask a later-window question after the exact occurrence has left the active reasoning window.
- Soul ablation: removing the relevant private memory should produce a measurable capability drop on Soul-dependent tasks.

## Governance boundary

Shared Field and Soul may use compatible mathematical token shapes inside the reasoning chamber, but they have different ownership and authority:

- Shared Field is canonical, Heart-governed, provenance-bearing public organism state.
- Soul is private per-Core state with crash-safe content-addressed lineage.
- Soul does not travel on proposal boards and is not visible to sibling Cores.
- Heart/Trainer may carry, hash, persist, validate lineage, and transport opaque Soul payloads as required by existing contracts, but do not treat Soul latents as canonical truth or shared semantic evidence.

The central proposal is therefore: **same reasoning mechanics, separate ingress, separate authority, separate persistence.** Private Soul tokens should sit beside the Core's learned field representations inside the reasoning chamber without becoming Shared Field.

## Immediate recommendation

Do not resume pointer/A0 training to test Soul. First specify the RNSC Soul interface and a minimal delayed-recall canary. The first Soul proof should demonstrate that a private memory written on one causal step can change a later answer when the current field no longer contains the needed evidence.
