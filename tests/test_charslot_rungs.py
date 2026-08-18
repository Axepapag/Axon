from __future__ import annotations

import argparse
import random

from training.trainer_slot import Phase0Curriculum, apply_charslot_rung_preset, draft_seed_for_mode


def test_partial_seed_fraction_is_configurable() -> None:
    assert draft_seed_for_mode("abcdefghij", "partial", partial_frac=0.7) == "abcdefg"
    assert draft_seed_for_mode("abcdefghij", "partial", partial_frac=0.3) == "abc"


def test_phase0_target_words_tightens_next_span() -> None:
    cur = Phase0Curriculum(max_chars=64, rng=random.Random(0), target_words=3)
    ex = cur._example_from_sentence("one two three four five six seven")
    assert ex["user_input"] == "one two three four"
    assert ex["answer"] == "five six seven"


def test_tight_stories_preset_sets_next_run_rungs() -> None:
    args = argparse.Namespace(
        charslot_rung_preset="tight-stories",
        charslot_mode_weights="0.3,0.3,0.4",
        charslot_partial_frac=0.5,
        phase0_target_words=0,
        text_corpus_weight=0.7,
        eval_n=32,
        history_chars=256,
    )
    apply_charslot_rung_preset(args)
    assert args.charslot_mode_weights == "0.10,0.55,0.35"
    assert args.charslot_partial_frac == 0.70
    assert args.phase0_target_words == 4
    assert args.text_corpus_weight == 0.90
    assert args.eval_n == 64
    assert args.history_chars == 256
