# Axon Memory Curriculum - Plain-Language Catalog (v0.1.0)

This is the teaching material for Axon's brain. It is a set of small, exact
exercises ("episodes") that any brain design can be tested with. The exercises
are made by deterministic generators: the same starting number (seed) always
makes the exact same exercise, so results can be checked and repeated.

Everything here uses only the 95 exact characters. Anything else is refused,
never fixed silently. A "WAIT" is a control signal, never an empty answer.

## How to use it (no programming needed)

The Lab will offer presets (buttons). Two exist now:

- **e0-first** (the starting set): small and understandable - copying,
  remembering after a pause, and knowing when to answer vs wait.
- **full-progression**: all eight exercise types, bigger and harder.

Each preset freezes its exercises into three separate piles before training:
**train** (practice), **validation** (checking during practice) and **test**
(only used at the end, never seen during practice). The piles are split by the
content itself (the facts and templates), not by row numbers, so nothing leaks
between piles. Every pile is stamped with a hash, so any later tampering is
detectable. If a pile cannot be filled exactly as designed, generation stops
with an error instead of quietly shipping a smaller set.

## The eight exercise types

1. **Copy** - see 1 to 8 characters, repeat them exactly. The first step.
2. **Delayed recall** - see characters, wait silently (0 to 32 quiet ticks),
   then repeat them. Tests memory over time.
3. **Distracted recall** - like delayed recall, but other things appear on the
   screen that must NOT be copied: random noise, look-alike characters, or the
   same decoy repeated. Tests focus.
4. **Key/value** - learn several "key is value" facts, then answer what one
   key holds. Tests association.
5. **Correction** - learn a fact, then hear it changed; answer with the NEW
   value, not the old one. Tests updating memory.
6. **Order and binding** - learn a short list or several pairs; answer with the
   Nth item, or which value belongs to which key, without swapping them.
7. **Response control** - learn WHEN to answer: hold (WAIT), answer exactly
   (COMMIT), or finish (END).
8. **Generalization** - combinations deliberately different from the others,
   kept in their own test pile, to measure real learning rather than
   memorization.

## How results are judged

- Exact answer right/wrong, and per-character accuracy.
- Binding errors (answered with a real stored fact, but the wrong one).
- Obsolete-memory errors (answered with the old value after a correction).
- WAIT/COMMIT/END correctness, and "wait confusion" (answered when it should
  have waited).
- Invalid-content count (any answer containing non-native characters fails).
- Reference numbers to compare against: a constant guesser, a most-common
  guesser, and chance rates - a brain must beat these honestly.

## Files

- `curriculum/schema.py` - the episode contract and the checker (fail closed).
- `curriculum/generators.py` - the eight deterministic exercise makers.
- `curriculum/splits.py` - the three frozen piles, hashes, and leak checks.
- `curriculum/presets.py` - the presets Jeff selects, and `materialize` which
  writes a preset's exercises + manifest + plain REPORT.txt to a folder.
- `curriculum/metrics.py` - scoring and the results record format.
- Materialized examples: `State/curriculum_v1/e0-first/` (git-ignored).

## Honest limits of v0.1.0

- Exercises are synthetic (made by rules), not real conversations.
- Capacity limits are reported explicitly, never hidden or called a pass.
- No big data dumps: generators are the product, previews stay bounded.
- The counterfactual memory tests (zero/swapped-state) and the real
  execution-path evaluations are designed but not yet executed through the
  Heart path.
