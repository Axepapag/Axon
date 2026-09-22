# Review: step-25 pointer-bootstrap geometry audit

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago
**Status:** read-only diagnosis; candidate remains stopped
**Candidate:** `english-candidate-1c991f8c911f79394e91`
**Accepted bundle / checkpoint / Soul:**
`ea681656e548c27afb28b5d59418da4313f87d6b0a4402344fa2478b83c3103e` /
`1b0282b026794109a5c574ed211f7c835ab05451e3785b2b7bf71d44d723cfe7` /
`13cf886a5a083200816c4487b12edbeb434500993dc8816ce6a374f0abe8f19e`

## Result

The native one-cell curriculum does not isolate the existing source-position
pointer. It combines three tasks: interpret an English decimal address,
generalize that address beyond the positions shown in training, and use the
learned query/key pointer to return the selected character. The one-step smoke
therefore could not answer whether the pointer itself is trainable.

No optimizer, checkpoint, Soul, Heart, field, trainer, or runtime state was
changed in this audit.

## Curriculum geometry

The declared positions are `0, 1, 7, 15, 31, 32, 33, 47, 63`, but the training
index formula reaches only six:

| Surface | Position counts |
| --- | --- |
| Train, 138 episodes | 0:23, 1:23, 15:23, 31:23, 33:23, 47:23 |
| Heldout, 24 episodes | 0:2, 1:3, 7:3, 15:3, 31:2, 32:2, 33:3, 47:3, 63:3 |

Positions 7, 32, and 63 are evaluation-only addresses. This is not an ordinary
heldout-context split; it asks the address selector to extrapolate to addresses
it was never trained to select.

The only accepted step consumed eight consecutive episodes because
`experiences_per_step=8`. They were `L@0`, `L@1`, `M@15`, `M@31`, `N@33`,
`N@47`, `P@0`, and `P@1`. A single accepted step therefore saw four target
characters and six addresses.

Every exercise identifies the source solely through prose of the form
`Return exactly the Cortex character at canonical position 47; stop.` The fresh
substrate candidate has not learned decimal-number meaning or the relation
between that sentence and a compiler address. The alignment loss supplies the
correct source during training, but the inference query still has to discover
that language-to-address mapping. This is a compound language-grounding task,
not a primitive pointer-only exercise.

The graduation gate also requires
`pointer_first_source_probability_mean >= 1.0`. A finite softmax can be
top-1 correct with a useful margin while remaining below exactly 1.0. Exact
top-1/output requirements are coherent; exact mean probability 1.0 is not a
useful finite-model graduation condition.

## Exact checkpoint trace

A CUDA replay loaded the verified step-25 checkpoint and candidate Soul. It
ran only the real FIRST surface and teacher-forced first pointer decision; no
training helper supplied a source address to the query.

- A balanced sample of two training episodes at each of the six reachable
  positions scored **0/12** pointer top-1.
- All 24 heldout FIRST phases scored **0/24** pointer top-1.
- Exact-source probabilities remained between approximately `0.00007` and
  `0.00061` when grouped by position.
- The pointer argmax repeatedly selected Cortex position 64 or 65.

Four counterfactuals changed only the decimal position in `user_input`, keeping
the Cortex and every other region fixed:

| Requested position change | Total variation in pointer distribution | Argmax before / after |
| --- | ---: | --- |
| 1 -> 7 | 0.000096 | cortex:64 / cortex:64 |
| 15 -> 31 | 0.000036 | cortex:65 / cortex:65 |
| 32 -> 47 | 0.000091 | cortex:65 / cortex:65 |
| 63 -> 33 | 0.000053 | cortex:64 / cortex:64 |

The exact-source probability also remained effectively unchanged across each
prompt swap. At this boundary the pointer query is insensitive to the requested
number; more repetitions of the same compound task would not be a controlled
test of the pointer primitive.

## Correction boundary

Keep step 25 as failed evidence and do not resume it. The cleaner parent for a
new governed intervention is the preserved step-24 bundle/checkpoint/Soul,
before the flawed curriculum update.

A replacement should separate capabilities:

1. First prove that the existing pointer can select a visibly and canonically
   marked Cortex cell across every declared position, with target characters,
   filler, and contexts held out independently. This is still the real Shared
   Field, compiler, pointer head, decoder, Heart phase order, and Soul path; the
   marker is readable field content, not a hidden teacher-only address.
2. Then teach symbolic address expressions, including decimal position text,
   as a separate grounding stage after the pointer primitive works.
3. Include every declared address on both training and heldout surfaces while
   holding out contexts and target characters rather than entire address
   classes.
4. Use an above-baseline smoke gate before any longer tranche. Retain exact
   top-1, exact first-scalar, valid nonempty output, and complete coverage for
   graduation. Treat exact-source probability as calibrated evidence or give
   it a ratified finite threshold below 1.0.
5. Add a prompt/marker-swap counterfactual so a fixed-address or
   prompt-insensitive pointer cannot pass.

These changes alter a ratified curriculum and gate. They require Jeff/table
ratification before implementation or another optimizer step.
