"""Tests for training/build_text_corpus.py sentence splitting (v2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.build_text_corpus import split_sentences


def test_plain_terminals():
    assert split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]


def test_closing_quote_after_terminal():
    # The v1 bug: 'friend?" The rabbit' was glued into one sentence.
    out = split_sentences('Do you want to be my friend?" The rabbit nodded.')
    assert out == ['Do you want to be my friend?"', "The rabbit nodded."]


def test_single_quote_after_terminal():
    out = split_sentences("He said 'stop.' Then he ran.")
    assert out == ["He said 'stop.'", "Then he ran."]


def test_no_split_mid_quote():
    out = split_sentences('She said, "lets go now" and smiled.')
    assert out == ['She said, "lets go now" and smiled.']
