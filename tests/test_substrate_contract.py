import numpy as np

import substrate  # noqa: E402 -- pythonpath is repo root per pyproject.toml


def test_default_alphabet_is_unique_and_excludes_empty():
    """default_alphabet() is the native letter bank: A-Z, a-z, 0-9, space,
    period, newline, !, ?. It does NOT include <empty>.

    <empty> is a LetterBank-level sentinel for null/blank slots, not a member
    of the default alphabet list.
    """
    alphabet = substrate.default_alphabet()

    assert "<empty>" not in alphabet
    assert len(alphabet) == len(set(alphabet))


def test_letter_bank_includes_empty():
    """LetterBank extends the default alphabet with <empty> as a null slot."""
    bank = substrate.get_letter_bank()

    assert "<empty>" in bank.chars
    assert bank.chars[-1] == "<empty>"
    assert len(bank.chars) == len(substrate.default_alphabet()) + 1


def test_writing_characters_roundtrip():
    """Every character in default_alphabet() round-trips exactly through
    cosine-nearest decoding (the LetterBank renderer)."""
    total, n_fail, failures = substrate.roundtrip_check()

    assert total > 0
    assert n_fail == 0
    assert failures == []


def test_character_slots_are_16d_nonzero_vectors():
    """Substrate slots are finite 16D vectors with stable nonzero norms.

    The 16D vectors are NOT unit-norm by design (norm ~3.45). The contract
    is: finite, 16-dimensional, nonzero, and cosine-roundtrip identity.
    """
    for ch in ["a", "Z", "0", " ", "\n", "<empty>"]:
        slot = substrate.char_to_slot(ch)

        assert slot.shape == (substrate.SLOT_DIM,)
        assert np.isfinite(slot).all()
        norm = float(np.linalg.norm(slot))
        assert norm > 1e-6, f"slot for {ch!r} has near-zero norm {norm}"


def test_slot_cosine_roundtrip():
    """Each character's slot, when decoded via cosine nearest-neighbor in the
    LetterBank, maps back to itself (or to <empty> for the empty slot)."""
    bank = substrate.get_letter_bank()

    for ch in substrate.default_alphabet():
        slot = substrate.char_to_slot(ch)
        decoded = bank.decode_letter(slot)
        assert decoded == ch, f"roundtrip failed for {ch!r} -> {decoded!r}"

    # <empty> slot decodes to empty string
    empty_slot = substrate.char_to_slot("<empty>")
    assert bank.decode_letter(empty_slot) == ""