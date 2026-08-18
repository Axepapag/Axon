from __future__ import annotations

import math

import pytest
import torch

from substrate import default_alphabet, get_letter_bank
from training.conversational_objective import (
    CHARACTER_COUNT,
    CLASS_COUNT,
    DECODER_WIDTH,
    EMPTY_INDEX,
    IGNORE_INDEX,
    build_conversational_targets,
    decode_parallel_indices,
    decode_parallel_logits,
    conversational_metrics,
    conversational_objective,
)


def _perfect_logits(texts: list[str]) -> torch.Tensor:
    bank = get_letter_bank()
    char_to_index = {char: index for index, char in enumerate(bank.chars)}
    logits = torch.full(
        (len(texts), DECODER_WIDTH, CLASS_COUNT),
        -20.0,
        dtype=torch.float32,
    )
    for row, text in enumerate(texts):
        for position, char in enumerate(text):
            logits[row, position, char_to_index[char]] = 20.0
        if len(text) < DECODER_WIDTH:
            logits[row, len(text), EMPTY_INDEX] = 20.0
        else:
            # Full-width decoding reaches length 64 because no position emits
            # the empty class.
            pass
        for position in range(len(text) + 1, DECODER_WIDTH):
            logits[row, position, 0] = 20.0
    return logits


def test_target_layout_uses_frozen_bank_terminator_and_ignored_tail() -> None:
    bank = get_letter_bank()
    assert len(default_alphabet()) == CHARACTER_COUNT
    assert len(bank.chars) == CLASS_COUNT
    assert bank.empty_index == EMPTY_INDEX

    targets = build_conversational_targets(
        ["Ab?", "", "x" * DECODER_WIDTH],
        answer_required=[True, False, True],
    )
    lookup = {char: index for index, char in enumerate(bank.chars)}

    assert targets.indices.shape == (3, DECODER_WIDTH)
    assert targets.indices[0, :3].tolist() == [
        lookup["A"],
        lookup["b"],
        lookup["?"],
    ]
    assert targets.indices[0, 3].item() == EMPTY_INDEX
    assert torch.all(targets.indices[0, 4:] == IGNORE_INDEX)

    assert targets.indices[1, 0].item() == EMPTY_INDEX
    assert torch.all(targets.indices[1, 1:] == IGNORE_INDEX)

    assert torch.all(targets.indices[2] == lookup["x"])
    assert targets.lengths.tolist() == [3, 0, DECODER_WIDTH]
    assert targets.has_termination_target.tolist() == [True, True, False]


def test_targets_fail_closed_for_empty_required_unsupported_and_overwidth() -> None:
    with pytest.raises(ValueError, match="requires an answer"):
        build_conversational_targets("", answer_required=True)
    with pytest.raises(ValueError, match="unsupported substrate characters"):
        build_conversational_targets("snowman \u2603")
    with pytest.raises(ValueError, match="exceeds decoder width"):
        build_conversational_targets("x" * (DECODER_WIDTH + 1))
    with pytest.raises(ValueError, match="must not overlap"):
        build_conversational_targets("x", ignore_index=EMPTY_INDEX)


def test_objective_matches_masked_ce_and_configurable_nonempty_auxiliary() -> None:
    torch.manual_seed(17)
    logits = torch.randn(2, DECODER_WIDTH, CLASS_COUNT, dtype=torch.float64)
    texts = ["Answer", ""]
    result = conversational_objective(
        logits,
        texts,
        answer_required=[True, False],
        first_position_nonempty_weight=0.75,
    )

    manual_ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, CLASS_COUNT),
        result.targets.indices.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    first = logits[0, 0]
    manual_aux = torch.nn.functional.softplus(
        first[EMPTY_INDEX] - torch.logsumexp(first[:CHARACTER_COUNT], dim=0)
    )
    assert torch.allclose(result.character_termination_loss, manual_ce)
    assert torch.allclose(result.first_position_nonempty_loss, manual_aux)
    assert torch.allclose(result.loss, manual_ce + 0.75 * manual_aux)


def test_ignored_tail_positions_have_exactly_zero_gradient() -> None:
    torch.manual_seed(23)
    logits = torch.randn(
        1,
        DECODER_WIDTH,
        CLASS_COUNT,
        dtype=torch.float64,
        requires_grad=True,
    )
    result = conversational_objective(
        logits,
        "Hi",
        answer_required=True,
        first_position_nonempty_weight=0.4,
    )
    result.loss.backward()

    assert logits.grad is not None
    # H, i, and the explicit terminator at position 2 are supervised.
    assert torch.count_nonzero(logits.grad[0, :3]).item() > 0
    # Every class at every position after that terminator is ignored.
    assert torch.count_nonzero(logits.grad[0, 3:]).item() == 0


def test_exact_length_metric_handles_termination_and_full_width() -> None:
    texts = ["Hello", "z" * DECODER_WIDTH]
    logits = _perfect_logits(texts)
    metrics = conversational_metrics(logits, texts)

    assert metrics.predicted.texts == tuple(texts)
    assert metrics.predicted.lengths == (5, DECODER_WIDTH)
    assert metrics.predicted.terminated == (True, False)
    assert metrics.exact_length_accuracy == 1.0
    assert metrics.exact_text_accuracy == 1.0

    early_stop = logits.clone()
    early_stop[1, 11, :] = -20.0
    early_stop[1, 11, EMPTY_INDEX] = 20.0
    metrics = conversational_metrics(early_stop, texts)
    assert metrics.predicted.lengths == (5, 11)
    assert metrics.exact_length_accuracy == 0.5
    assert metrics.exact_text_accuracy == 0.5


def test_batch_construction_and_loss_are_deterministic() -> None:
    generator = torch.Generator().manual_seed(101)
    logits = torch.randn(
        3,
        DECODER_WIDTH,
        CLASS_COUNT,
        generator=generator,
        dtype=torch.float32,
    )
    texts = ["One", "Two lines\n", ""]
    required = [True, True, False]

    first = conversational_objective(
        logits,
        texts,
        answer_required=required,
        first_position_nonempty_weight=0.25,
    )
    second = conversational_objective(
        logits,
        texts,
        answer_required=required,
        first_position_nonempty_weight=0.25,
    )

    assert torch.equal(first.targets.indices, second.targets.indices)
    assert torch.equal(first.targets.lengths, second.targets.lengths)
    assert torch.equal(first.loss, second.loss)
    assert math.isfinite(float(first.loss))


def test_no_required_answers_use_an_exact_finite_zero_auxiliary() -> None:
    logits = torch.full(
        (1, DECODER_WIDTH, CLASS_COUNT),
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
    )
    result = conversational_objective(
        logits,
        "",
        answer_required=False,
        first_position_nonempty_weight=0.0,
    )

    assert result.first_position_nonempty_loss.item() == 0.0
    assert torch.isfinite(result.character_termination_loss)
    assert torch.isfinite(result.loss)


def test_finite_logits_that_overflow_a_loss_fail_closed() -> None:
    logits = torch.full(
        (1, DECODER_WIDTH, CLASS_COUNT),
        -torch.finfo(torch.float32).max,
        dtype=torch.float32,
    )
    logits[:, :, EMPTY_INDEX] = torch.finfo(torch.float32).max

    with pytest.raises(FloatingPointError, match="loss is non-finite"):
        conversational_objective(
            logits,
            "A",
            answer_required=True,
            first_position_nonempty_weight=1.0,
        )


@pytest.mark.parametrize(
    "logits, expected_exception",
    [
        (torch.zeros(DECODER_WIDTH, CLASS_COUNT), ValueError),
        (torch.zeros(1, DECODER_WIDTH, CLASS_COUNT, dtype=torch.long), TypeError),
        (
            torch.full(
                (1, DECODER_WIDTH, CLASS_COUNT),
                float("nan"),
                dtype=torch.float32,
            ),
            ValueError,
        ),
        (
            torch.full(
                (1, DECODER_WIDTH, CLASS_COUNT),
                float("inf"),
                dtype=torch.float32,
            ),
            ValueError,
        ),
    ],
)
def test_objective_rejects_invalid_logits(
    logits: torch.Tensor,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        conversational_objective(logits, "A")


def test_decoders_reject_invalid_shape_dtype_values_and_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="shape"):
        decode_parallel_indices(torch.zeros(DECODER_WIDTH, dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        decode_parallel_indices(
            torch.zeros(1, DECODER_WIDTH, dtype=torch.float32)
        )
    invalid_index = torch.zeros(1, DECODER_WIDTH, dtype=torch.long)
    invalid_index[0, 0] = CLASS_COUNT
    with pytest.raises(ValueError, match="outside"):
        decode_parallel_indices(invalid_index)

    with pytest.raises(ValueError, match="shape"):
        decode_parallel_logits(torch.zeros(1, DECODER_WIDTH, CLASS_COUNT - 1))
    with pytest.raises(TypeError, match="floating"):
        decode_parallel_logits(
            torch.zeros(1, DECODER_WIDTH, CLASS_COUNT, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="non-finite"):
        decode_parallel_logits(
            torch.full(
                (1, DECODER_WIDTH, CLASS_COUNT),
                float("nan"),
                dtype=torch.float32,
            )
        )
