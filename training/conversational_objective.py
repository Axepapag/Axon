"""Isolated loss and metrics for Axon's 64-position parallel text decoder.

The decoder has 96 classes at every position: the frozen 95-character
substrate alphabet followed by ``<empty>`` at index 95.  For a target of
length ``L < 64``, positions ``0..L-1`` contain character targets, position
``L`` contains the explicit empty/termination target, and every later
position uses ``IGNORE_INDEX``.  A full-width target (``L == 64``) has no
spare termination position, so all 64 character positions are supervised;
decoding assigns length 64 when no empty class is predicted.

This module intentionally has no trainer or runtime dependencies.  It can be
integrated only after its objective and free-running metrics have passed their
own tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from substrate import assert_supported_text, default_alphabet, get_letter_bank


DECODER_WIDTH = 64
CHARACTER_COUNT = 95
CLASS_COUNT = 96
EMPTY_INDEX = 95
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ConversationalTargets:
    """A deterministic batch of masked parallel-decoder targets."""

    indices: torch.Tensor
    lengths: torch.Tensor
    answer_required: torch.Tensor
    has_termination_target: torch.Tensor


@dataclass(frozen=True)
class ConversationalObjectiveResult:
    """Loss components and the exact target tensor used to compute them."""

    loss: torch.Tensor
    character_termination_loss: torch.Tensor
    first_position_nonempty_loss: torch.Tensor
    targets: ConversationalTargets


@dataclass(frozen=True)
class ParallelDecodeResult:
    """Text decoded by stopping at the first explicit empty class."""

    texts: tuple[str, ...]
    lengths: tuple[int, ...]
    terminated: tuple[bool, ...]


@dataclass(frozen=True)
class ConversationalMetrics:
    """Deterministic, no-gold-in-decoder metrics for a completed batch."""

    exact_length_accuracy: float
    exact_text_accuracy: float
    predicted: ParallelDecodeResult
    target_lengths: tuple[int, ...]


def _validated_bank_chars() -> tuple[str, ...]:
    """Return the frozen class order, failing closed if its contract drifted."""

    alphabet = tuple(default_alphabet())
    bank = get_letter_bank()
    bank_chars = tuple(bank.chars)
    if len(alphabet) != CHARACTER_COUNT:
        raise RuntimeError(
            f"substrate alphabet has {len(alphabet)} characters; "
            f"expected {CHARACTER_COUNT}"
        )
    if len(set(alphabet)) != CHARACTER_COUNT:
        raise RuntimeError("substrate alphabet contains duplicate characters")
    if bank.empty_index != EMPTY_INDEX:
        raise RuntimeError(
            f"letter-bank empty index is {bank.empty_index}; "
            f"expected {EMPTY_INDEX}"
        )
    if len(bank_chars) != CLASS_COUNT:
        raise RuntimeError(
            f"letter bank has {len(bank_chars)} classes; expected {CLASS_COUNT}"
        )
    if bank_chars[:CHARACTER_COUNT] != alphabet:
        raise RuntimeError("letter-bank character order differs from the substrate")
    if bank_chars[EMPTY_INDEX] != "<empty>":
        raise RuntimeError("letter-bank class 95 is not <empty>")
    return bank_chars


def _normalize_texts(texts: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, str):
        result = (texts,)
    else:
        result = tuple(texts)
    if not result:
        raise ValueError("conversational target batch must not be empty")
    for index, text in enumerate(result):
        if not isinstance(text, str):
            raise TypeError(f"target {index} must be a string")
    return result


def _normalize_answer_required(
    answer_required: bool | Sequence[bool],
    batch_size: int,
) -> tuple[bool, ...]:
    if isinstance(answer_required, bool):
        return (answer_required,) * batch_size
    flags = tuple(answer_required)
    if len(flags) != batch_size:
        raise ValueError(
            "answer_required must be one bool or have one entry per target"
        )
    if any(not isinstance(flag, bool) for flag in flags):
        raise TypeError("answer_required entries must be bool values")
    return flags


def build_conversational_targets(
    texts: str | Sequence[str],
    *,
    answer_required: bool | Sequence[bool] = True,
    device: torch.device | str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> ConversationalTargets:
    """Encode targets using characters, one terminator, then ignored tail.

    Empty targets are valid only for examples explicitly marked as not
    answer-required.  Text longer than 64 characters and unsupported substrate
    characters are rejected rather than clipped or substituted.
    """

    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")
    if 0 <= ignore_index < CLASS_COUNT:
        raise ValueError("ignore_index must not overlap a decoder class")
    bank_chars = _validated_bank_chars()
    values = _normalize_texts(texts)
    required = _normalize_answer_required(answer_required, len(values))
    char_to_index = {
        character: index
        for index, character in enumerate(bank_chars[:CHARACTER_COUNT])
    }

    indices = torch.full(
        (len(values), DECODER_WIDTH),
        int(ignore_index),
        dtype=torch.long,
        device=device,
    )
    lengths = torch.empty(len(values), dtype=torch.long, device=device)
    required_tensor = torch.tensor(required, dtype=torch.bool, device=device)
    has_termination = torch.empty(len(values), dtype=torch.bool, device=device)

    for row, (text, is_required) in enumerate(zip(values, required, strict=True)):
        assert_supported_text(text)
        length = len(text)
        if length > DECODER_WIDTH:
            raise ValueError(
                f"target {row} length {length} exceeds decoder width "
                f"{DECODER_WIDTH}"
            )
        if is_required and length == 0:
            raise ValueError(
                f"target {row} is empty but the example requires an answer"
            )
        for position, character in enumerate(text):
            try:
                indices[row, position] = char_to_index[character]
            except KeyError as exc:  # defensive if the bank contract changes
                raise ValueError(
                    f"target {row} contains unsupported character {character!r}"
                ) from exc
        if length < DECODER_WIDTH:
            indices[row, length] = EMPTY_INDEX
            has_termination[row] = True
        else:
            # Width 64 has no position 65 in which to place a terminator.
            has_termination[row] = False
        lengths[row] = length

    return ConversationalTargets(
        indices=indices,
        lengths=lengths,
        answer_required=required_tensor,
        has_termination_target=has_termination,
    )


def conversational_objective(
    logits: torch.Tensor,
    texts: str | Sequence[str],
    *,
    answer_required: bool | Sequence[bool] = True,
    first_position_nonempty_weight: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> ConversationalObjectiveResult:
    """Compute masked CE plus an optional answer-required nonempty penalty.

    The auxiliary term is ``-log(1 - p(empty_at_position_0))`` averaged only
    over answer-required examples.  It is zero when the batch has no such
    examples.  The function contains no randomness and preserves input order.
    """

    if logits.ndim != 3 or tuple(logits.shape[1:]) != (
        DECODER_WIDTH,
        CLASS_COUNT,
    ):
        raise ValueError(
            "logits must have shape "
            f"(batch, {DECODER_WIDTH}, {CLASS_COUNT}); got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits contain non-finite values")
    if (
        not math.isfinite(first_position_nonempty_weight)
        or first_position_nonempty_weight < 0.0
    ):
        raise ValueError(
            "first_position_nonempty_weight must be finite and non-negative"
        )

    values = _normalize_texts(texts)
    if logits.shape[0] != len(values):
        raise ValueError(
            f"logits batch size {logits.shape[0]} does not match "
            f"{len(values)} targets"
        )
    targets = build_conversational_targets(
        values,
        answer_required=answer_required,
        device=logits.device,
        ignore_index=ignore_index,
    )
    character_termination_loss = F.cross_entropy(
        logits.reshape(-1, CLASS_COUNT),
        targets.indices.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    if not torch.isfinite(character_termination_loss).item():
        raise FloatingPointError(
            "character/termination loss is non-finite for finite input logits"
        )

    required_logits = logits[targets.answer_required, 0, :]
    if required_logits.shape[0]:
        empty_logits = required_logits[:, EMPTY_INDEX]
        nonempty_logsumexp = torch.logsumexp(
            required_logits[:, :CHARACTER_COUNT],
            dim=-1,
        )
        first_position_nonempty_loss = F.softplus(
            empty_logits - nonempty_logsumexp
        ).mean()
    else:
        # Do not derive zero from a reduction of extreme finite logits:
        # a finite sum may overflow to infinity, and infinity * 0 is NaN.
        first_position_nonempty_loss = logits.new_zeros(())
    if not torch.isfinite(first_position_nonempty_loss).item():
        raise FloatingPointError(
            "first-position nonempty loss is non-finite for finite input logits"
        )

    loss = (
        character_termination_loss
        + float(first_position_nonempty_weight) * first_position_nonempty_loss
    )
    if not torch.isfinite(loss).item():
        raise FloatingPointError(
            "combined conversational loss is non-finite for finite input logits"
        )
    return ConversationalObjectiveResult(
        loss=loss,
        character_termination_loss=character_termination_loss,
        first_position_nonempty_loss=first_position_nonempty_loss,
        targets=targets,
    )


def decode_parallel_indices(indices: torch.Tensor) -> ParallelDecodeResult:
    """Decode ``(batch, 64)`` class indices through first-empty termination."""

    bank_chars = _validated_bank_chars()
    if indices.ndim != 2 or indices.shape[1] != DECODER_WIDTH:
        raise ValueError(
            f"indices must have shape (batch, {DECODER_WIDTH}); "
            f"got {tuple(indices.shape)}"
        )
    if indices.shape[0] == 0:
        raise ValueError("decoder batch must not be empty")
    if indices.dtype == torch.bool or indices.is_floating_point():
        raise TypeError("indices must use an integer dtype")

    texts: list[str] = []
    lengths: list[int] = []
    terminated: list[bool] = []
    for row, raw_values in enumerate(indices.detach().cpu().tolist()):
        characters: list[str] = []
        row_terminated = False
        for position, raw_index in enumerate(raw_values):
            index = int(raw_index)
            if not 0 <= index < CLASS_COUNT:
                raise ValueError(
                    f"decoder index at row {row}, position {position} is "
                    f"outside [0, {CLASS_COUNT - 1}]: {index}"
                )
            if index == EMPTY_INDEX:
                row_terminated = True
                break
            characters.append(bank_chars[index])
        text = "".join(characters)
        texts.append(text)
        lengths.append(len(text))
        terminated.append(row_terminated)
    return ParallelDecodeResult(
        texts=tuple(texts),
        lengths=tuple(lengths),
        terminated=tuple(terminated),
    )


def decode_parallel_logits(logits: torch.Tensor) -> ParallelDecodeResult:
    """Argmax-decode a deterministic batch of 96-class parallel logits."""

    if logits.ndim != 3 or tuple(logits.shape[1:]) != (
        DECODER_WIDTH,
        CLASS_COUNT,
    ):
        raise ValueError(
            "logits must have shape "
            f"(batch, {DECODER_WIDTH}, {CLASS_COUNT}); got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits contain non-finite values")
    return decode_parallel_indices(logits.argmax(dim=-1))


def conversational_metrics(
    logits: torch.Tensor,
    texts: str | Sequence[str],
) -> ConversationalMetrics:
    """Report exact decoded length and exact text accuracy after argmax."""

    values = _normalize_texts(texts)
    for row, text in enumerate(values):
        assert_supported_text(text)
        if len(text) > DECODER_WIDTH:
            raise ValueError(
                f"target {row} length {len(text)} exceeds decoder width "
                f"{DECODER_WIDTH}"
            )
    predicted = decode_parallel_logits(logits)
    if logits.shape[0] != len(values):
        raise ValueError(
            f"logits batch size {logits.shape[0]} does not match "
            f"{len(values)} targets"
        )
    target_lengths = tuple(len(text) for text in values)
    length_matches = [
        int(predicted_length == target_length)
        for predicted_length, target_length in zip(
            predicted.lengths,
            target_lengths,
            strict=True,
        )
    ]
    text_matches = [
        int(predicted_text == target_text)
        for predicted_text, target_text in zip(
            predicted.texts,
            values,
            strict=True,
        )
    ]
    return ConversationalMetrics(
        exact_length_accuracy=sum(length_matches) / len(length_matches),
        exact_text_accuracy=sum(text_matches) / len(text_matches),
        predicted=predicted,
        target_lengths=target_lengths,
    )


__all__ = [
    "DECODER_WIDTH",
    "CHARACTER_COUNT",
    "CLASS_COUNT",
    "EMPTY_INDEX",
    "IGNORE_INDEX",
    "ConversationalTargets",
    "ConversationalObjectiveResult",
    "ParallelDecodeResult",
    "ConversationalMetrics",
    "build_conversational_targets",
    "conversational_objective",
    "decode_parallel_indices",
    "decode_parallel_logits",
    "conversational_metrics",
]
