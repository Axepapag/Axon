"""Read-only causal diagnostics for the D64 exact-source pointer.

The oracle conditions in this module are evaluator interventions.  They do not
modify model parameters, Soul, canonical State, or the production decoder and
must never be counted as learned competence.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from substrate import encode_unicode_text

from .living_reasoning_d64 import LivingReasoningCoreD64, LivingReasoningForward


def mix_living_probabilities(
    generated_probabilities: torch.Tensor,
    copied_probabilities: torch.Tensor,
    generate_probability: torch.Tensor | float,
    *,
    eos_index: int,
) -> torch.Tensor:
    """Reproduce the living decoder's content mixture and ordinary EOS route."""

    if generated_probabilities.shape != copied_probabilities.shape:
        raise ValueError("generated and copied distributions must have the same shape")
    if generated_probabilities.ndim != 1:
        raise ValueError("first-decision distributions must be rank one")
    if eos_index < 1 or eos_index >= generated_probabilities.numel():
        raise ValueError("eos_index is outside the decoder distribution")
    generate = torch.as_tensor(
        generate_probability,
        device=generated_probabilities.device,
        dtype=generated_probabilities.dtype,
    )
    if generate.numel() != 1:
        raise ValueError("generate_probability must be scalar")
    if not bool(torch.isfinite(generate).all()) or not 0.0 <= float(generate.item()) <= 1.0:
        raise ValueError("generate_probability must be finite in [0, 1]")

    base = generate * generated_probabilities + (1.0 - generate) * copied_probabilities
    generated_eos = generated_probabilities[eos_index]
    base_eos = base[eos_index]
    tiny = torch.finfo(base.dtype).tiny
    epsilon = torch.finfo(base.dtype).eps
    generated_eos = generated_eos.clamp(min=tiny, max=1.0 - epsilon)
    base_eos = base_eos.clamp(min=tiny, max=1.0 - epsilon)
    content_scale = (1.0 - generated_eos) / (1.0 - base_eos)
    result = base.clone()
    result[:eos_index] = result[:eos_index] * content_scale
    result[eos_index] = generated_eos
    return result


@torch.no_grad()
def first_decision_diagnostic(
    model: LivingReasoningCoreD64,
    output: LivingReasoningForward,
    *,
    target_text: str,
    alignment_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure normal, oracle-pointer, and oracle-pointer-plus-copy behavior."""

    if not target_text:
        raise ValueError("target_text must be nonempty")
    decoded = model.decode_teacher(
        output.reader_state,
        target_text,
        head=1,
        memory=output.complete_memory,
        return_alignment=True,
    )
    normal_log_probabilities, _targets, decoder_alignment = decoded
    supervised = model.alignment_supervision(
        target_text=target_text,
        memory=output.complete_memory,
        decoder_alignment=decoder_alignment,
        specification=alignment_specification,
    )
    source_indices = tuple(supervised["source_memory_indices"])
    if not source_indices:
        raise ValueError("alignment does not identify a first source cell")
    expected_source = int(source_indices[0])
    expected_transport = int(encode_unicode_text(target_text)[0])
    observed_transport = int(output.complete_memory.char_indices[0, expected_source].item())
    if observed_transport != expected_transport:
        raise RuntimeError("receipt-certified source does not contain the expected transport unit")

    position_logits = decoder_alignment["position_logits"][0, 0]
    pointer = F.softmax(position_logits, dim=-1)
    valid_sources = output.complete_memory.char_indices[0].ge(0)
    pointer_top = int(pointer.argmax(dim=-1).item())
    other_logits = position_logits[valid_sources].clone()
    valid_indices = torch.nonzero(valid_sources, as_tuple=False).flatten()
    expected_valid_offset = int(
        torch.nonzero(valid_indices.eq(expected_source), as_tuple=False).item()
    )
    other_logits[expected_valid_offset] = -torch.inf
    pointer_margin = float(
        (position_logits[expected_source] - other_logits.max()).item()
    )
    pointer_probability = float(pointer[expected_source].item())

    generated = F.softmax(decoder_alignment["generated_logits"][0, 0], dim=-1)
    normal_generate = torch.sigmoid(decoder_alignment["generate_gate_logits"][0, 0])
    copied = torch.zeros_like(generated)
    copied[observed_transport] = 1.0
    oracle_normal_route = mix_living_probabilities(
        generated,
        copied,
        normal_generate,
        eos_index=model.eos_index,
    )
    oracle_forced_copy = mix_living_probabilities(
        generated,
        copied,
        0.0,
        eos_index=model.eos_index,
    )
    normal = normal_log_probabilities[0, 0].exp()

    def condition(probabilities: torch.Tensor) -> dict[str, Any]:
        top = int(probabilities.argmax(dim=-1).item())
        content_top = int(probabilities[: model.eos_index].argmax(dim=-1).item())
        return {
            "top1_token": top,
            "top1_correct": top == expected_transport,
            "content_top1_token": content_top,
            "content_top1_correct": content_top == expected_transport,
            "target_probability": float(probabilities[expected_transport].item()),
            "eos_probability": float(probabilities[model.eos_index].item()),
            "eos_top1": top == model.eos_index,
            "probability_sum": float(probabilities.sum().item()),
        }

    fused = decoder_alignment["decoder_fused"][0, 0]
    query = model.position_query(fused)
    keys = model.position_key(output.complete_memory.states)[0]
    cortex_keys: dict[int, torch.Tensor] = {}
    for index, receipt in enumerate(output.complete_memory.receipts):
        if receipt is None or receipt.address.region.value != "cortex":
            continue
        position = int(receipt.address.region_position)
        if position in cortex_keys:
            raise RuntimeError("canonical Cortex position appears more than once in memory")
        cortex_keys[position] = keys[index].detach().cpu()

    return {
        "expected_source_memory_index": expected_source,
        "expected_cortex_position": int(
            output.complete_memory.receipts[expected_source].address.region_position
        ),
        "expected_transport_token": expected_transport,
        "pointer_top1_memory_index": pointer_top,
        "pointer_top1_correct": pointer_top == expected_source,
        "pointer_probability": pointer_probability,
        "pointer_nll": -math.log(max(pointer_probability, torch.finfo(pointer.dtype).tiny)),
        "pointer_logit_margin": pointer_margin,
        "generate_route_probability": float(normal_generate.item()),
        "copy_route_probability": float((1.0 - normal_generate).item()),
        "normal": condition(normal),
        "oracle_pointer_normal_route": condition(oracle_normal_route),
        "oracle_pointer_forced_copy": condition(oracle_forced_copy),
        "reader_vector": output.reader_state.mean(dim=1)[0].detach().cpu(),
        "decoder_fused_vector": fused.detach().cpu(),
        "pointer_query_vector": query.detach().cpu(),
        "pointer_distribution": pointer.detach().cpu(),
        "cortex_keys": cortex_keys,
        "soul_telemetry": dict(output.soul_telemetry),
    }


def summarize_vectors(vectors: Sequence[torch.Tensor]) -> dict[str, float]:
    """Summarize variation without exposing full latent tensors as evidence."""

    if not vectors:
        raise ValueError("at least one vector is required")
    matrix = torch.stack([item.detach().to(dtype=torch.float64, device="cpu") for item in vectors])
    if matrix.ndim != 2:
        raise ValueError("vectors must be rank one and shape-compatible")
    variance = matrix.var(dim=0, unbiased=False)
    if matrix.shape[0] > 1:
        distances = torch.pdist(matrix, p=2)
        mean_pairwise = float(distances.mean().item())
        minimum_pairwise = float(distances.min().item())
    else:
        mean_pairwise = 0.0
        minimum_pairwise = 0.0
    return {
        "sample_count": float(matrix.shape[0]),
        "dimension_count": float(matrix.shape[1]),
        "mean_per_dimension_variance": float(variance.mean().item()),
        "maximum_per_dimension_variance": float(variance.max().item()),
        "mean_vector_l2": float(matrix.norm(dim=1).mean().item()),
        "mean_pairwise_l2": mean_pairwise,
        "minimum_pairwise_l2": minimum_pairwise,
    }


def leave_one_group_out_centroid_accuracy(
    vectors: Sequence[torch.Tensor],
    labels: Sequence[int],
    groups: Sequence[str],
) -> dict[str, float]:
    """Measure whether a representation consistently separates address labels."""

    if not vectors or not (len(vectors) == len(labels) == len(groups)):
        raise ValueError("vectors, labels, and groups must be nonempty and aligned")
    matrix = F.normalize(
        torch.stack([item.detach().to(dtype=torch.float64, device="cpu") for item in vectors]),
        dim=-1,
    )
    unique_labels = tuple(sorted(set(int(item) for item in labels)))
    correct = 0
    eligible = 0
    for index, vector in enumerate(matrix):
        centroids: list[torch.Tensor] = []
        candidates: list[int] = []
        for label in unique_labels:
            train_indices = [
                other
                for other, other_label in enumerate(labels)
                if int(other_label) == label and groups[other] != groups[index]
            ]
            if not train_indices:
                continue
            centroid = F.normalize(matrix[train_indices].mean(dim=0), dim=0)
            centroids.append(centroid)
            candidates.append(label)
        if int(labels[index]) not in candidates:
            continue
        scores = torch.stack(centroids) @ vector
        predicted = candidates[int(scores.argmax().item())]
        correct += int(predicted == int(labels[index]))
        eligible += 1
    return {
        "eligible_count": float(eligible),
        "correct_count": float(correct),
        "accuracy": correct / max(1, eligible),
        "label_count": float(len(unique_labels)),
        "uniform_chance": 1.0 / len(unique_labels),
    }


def cross_episode_cortex_key_accuracy(
    episode_keys: Sequence[tuple[str, Mapping[int, torch.Tensor]]],
) -> dict[str, float]:
    """Classify every Cortex position from keys learned on other episodes."""

    if len(episode_keys) < 2:
        raise ValueError("cross-episode key geometry requires at least two episodes")
    positions = tuple(sorted(set.intersection(*(set(keys) for _, keys in episode_keys))))
    if not positions:
        raise ValueError("episodes have no common Cortex positions")
    correct = 0
    count = 0
    for heldout_name, heldout in episode_keys:
        centroids: list[torch.Tensor] = []
        for position in positions:
            training = [
                keys[position].detach().to(dtype=torch.float64, device="cpu")
                for name, keys in episode_keys
                if name != heldout_name
            ]
            centroids.append(F.normalize(torch.stack(training).mean(dim=0), dim=0))
        centroid_matrix = torch.stack(centroids)
        for position in positions:
            query = F.normalize(
                heldout[position].detach().to(dtype=torch.float64, device="cpu"),
                dim=0,
            )
            predicted = positions[int((centroid_matrix @ query).argmax().item())]
            correct += int(predicted == position)
            count += 1
    return {
        "eligible_count": float(count),
        "correct_count": float(correct),
        "accuracy": correct / max(1, count),
        "position_count": float(len(positions)),
        "uniform_chance": 1.0 / len(positions),
    }


def aggregate_first_decisions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar D0 outcomes while keeping oracle results distinct."""

    items = tuple(rows)
    if not items:
        raise ValueError("at least one first-decision row is required")

    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for row in items:
            value: Any = row
            for component in path:
                value = value[component]
            values.append(float(value))
        return sum(values) / len(values)

    conditions = {}
    for name in ("normal", "oracle_pointer_normal_route", "oracle_pointer_forced_copy"):
        conditions[name] = {
            "top1_accuracy": mean((name, "top1_correct")),
            "content_top1_accuracy": mean((name, "content_top1_correct")),
            "target_probability_mean": mean((name, "target_probability")),
            "eos_top1_rate": mean((name, "eos_top1")),
            "eos_probability_mean": mean((name, "eos_probability")),
            "probability_sum_mean": mean((name, "probability_sum")),
        }
    return {
        "sample_count": len(items),
        "pointer_top1_accuracy": mean(("pointer_top1_correct",)),
        "pointer_probability_mean": mean(("pointer_probability",)),
        "pointer_nll_mean": mean(("pointer_nll",)),
        "pointer_logit_margin_mean": mean(("pointer_logit_margin",)),
        "copy_route_probability_mean": mean(("copy_route_probability",)),
        "conditions": conditions,
    }


__all__ = [
    "aggregate_first_decisions",
    "cross_episode_cortex_key_accuracy",
    "first_decision_diagnostic",
    "leave_one_group_out_centroid_accuracy",
    "mix_living_probabilities",
    "summarize_vectors",
]
