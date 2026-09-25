"""B4 curriculum, evaluation, and real Trainer preflight for the D512 continuous Core."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from runtime.axon_runtime import ContinuousCoreD512
from runtime.field import (
    D16_VIEW_SCHEMA,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
    materialize_d16_view,
)
from runtime.trainer import (
    CompleteFieldTrainingContract,
    DeclaredTrainingBound,
    ParameterInventory,
    ParameterMutationPlanLike,
    PreflightEvidenceKind,
    TrainingBoundCategory,
    TrainingPreflightEvidence,
    TrainingPreflightReceipt,
    build_training_preflight_receipt,
)
from substrate import (
    NATIVE_TOKEN_COUNT,
    TRANSPORT_VOCAB_SIZE,
    UNICODE_TRANSPORT_SCHEMA,
    decode_unicode_tokens,
    default_alphabet,
    encode_unicode_text,
    transport_token_cell16,
    unicode_transport_bank,
)

D512_COPY_CURRICULUM_SCHEMA = "axon-d512-substrate-copy-curriculum-v1"
D512_COPY_OBJECTIVE_SCHEMA = "axon-d512-variable-copy-objective-v1"


@dataclass(frozen=True, slots=True)
class D512CopyCase:
    split: str
    text: str
    token_ids: tuple[int, ...]
    field_id: str
    view_id: str
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout"}:
            raise ValueError("copy case split must be train or heldout")
        if not self.text:
            raise ValueError("copy case text must be nonempty")
        tokens = tuple(int(item) for item in self.token_ids)
        if not tokens or any(item < 0 or item >= TRANSPORT_VOCAB_SIZE for item in tokens):
            raise ValueError("copy case contains invalid transport token")
        if decode_unicode_tokens(tokens) != self.text:
            raise ValueError("copy case transport does not roundtrip exactly")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(
            self,
            "case_id",
            canonical_sha256(
                {
                    "schema": D512_COPY_CURRICULUM_SCHEMA,
                    "split": self.split,
                    "text": self.text,
                    "token_ids": list(tokens),
                    "field_id": self.field_id,
                    "view_id": self.view_id,
                }
            ),
        )

    @property
    def transport_length(self) -> int:
        return len(self.token_ids)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "field_id": self.field_id,
            "view_id": self.view_id,
        }


@dataclass(frozen=True, slots=True)
class D512CopyCurriculum:
    train_cases: tuple[D512CopyCase, ...]
    heldout_cases: tuple[D512CopyCase, ...]
    train_manifest_id: str = field(init=False)
    heldout_manifest_id: str = field(init=False)
    curriculum_id: str = field(init=False)

    def __post_init__(self) -> None:
        train = tuple(self.train_cases)
        heldout = tuple(self.heldout_cases)
        if not train or not heldout:
            raise ValueError("copy curriculum needs nonempty train and heldout cases")
        train_text = {item.text for item in train}
        heldout_text = {item.text for item in heldout}
        if train_text & heldout_text:
            raise ValueError("copy curriculum train and heldout text must be disjoint")
        if any(item.split != "train" for item in train) or any(item.split != "heldout" for item in heldout):
            raise ValueError("copy curriculum case split mismatch")
        object.__setattr__(self, "train_cases", train)
        object.__setattr__(self, "heldout_cases", heldout)
        train_manifest = canonical_sha256({"split": "train", "case_ids": [item.case_id for item in train]})
        heldout_manifest = canonical_sha256({"split": "heldout", "case_ids": [item.case_id for item in heldout]})
        object.__setattr__(self, "train_manifest_id", train_manifest)
        object.__setattr__(self, "heldout_manifest_id", heldout_manifest)
        object.__setattr__(
            self,
            "curriculum_id",
            canonical_sha256(
                {
                    "schema": D512_COPY_CURRICULUM_SCHEMA,
                    "train_manifest_id": train_manifest,
                    "heldout_manifest_id": heldout_manifest,
                }
            ),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": D512_COPY_CURRICULUM_SCHEMA,
            "curriculum_id": self.curriculum_id,
            "train_manifest_id": self.train_manifest_id,
            "heldout_manifest_id": self.heldout_manifest_id,
            "train_cases": [item.to_canonical_dict() for item in self.train_cases],
            "heldout_cases": [item.to_canonical_dict() for item in self.heldout_cases],
        }

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root).resolve() / "training" / "continuous_core_d512" / "curricula"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.curriculum_id}.json"
        payload = canonical_json_bytes(self.to_canonical_dict())
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError("D512 curriculum identity collision")
            return path
        path.write_bytes(payload)
        return path


@dataclass(frozen=True, slots=True)
class D512CopyEvaluation:
    cases: int
    content_tokens: int
    teacher_content_accuracy: float
    teacher_eos_accuracy: float
    free_exact_accuracy: float
    termination_accuracy: float
    valid_unicode_accuracy: float
    strongest_constant_floor: float
    strongest_content_constant_floor: float
    mean_loss: float
    evaluation_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": "axon-d512-copy-evaluation-v1",
            "cases": self.cases,
            "content_tokens": self.content_tokens,
            "teacher_content_accuracy": self.teacher_content_accuracy,
            "teacher_eos_accuracy": self.teacher_eos_accuracy,
            "free_exact_accuracy": self.free_exact_accuracy,
            "termination_accuracy": self.termination_accuracy,
            "valid_unicode_accuracy": self.valid_unicode_accuracy,
            "strongest_constant_floor": self.strongest_constant_floor,
            "strongest_content_constant_floor": self.strongest_content_constant_floor,
            "mean_loss": self.mean_loss,
        }
        if include_id:
            value["evaluation_id"] = self.evaluation_id
        return value


def _case(split: str, text: str, ordinal: int) -> D512CopyCase:
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text}, tick_id=ordinal + 1)
    view = materialize_d16_view(snapshot)
    token_ids = encode_unicode_text(text)
    materialized = tuple(
        int(token_id)
        for region in view.regions
        for token_id in region.token_ids
    )
    if materialized != token_ids:
        raise RuntimeError("full D16 view does not reproduce exact USER_INPUT transport")
    return D512CopyCase(
        split=split,
        text=text,
        token_ids=token_ids,
        field_id=snapshot.field_id,
        view_id=view.view_id,
    )


def _unique_random_texts(*, seed: int, count: int, forbidden: set[str]) -> list[str]:
    rng = random.Random(seed)
    alphabet = tuple(default_alphabet())
    unicode_atoms = ("λ", "é", "中", "🙂", "🧠", "\t", "\x00")
    values: list[str] = []
    seen = set(forbidden)
    while len(values) < count:
        length = rng.randint(1, 8)
        chars: list[str] = []
        for _ in range(length):
            if rng.random() < 0.08:
                chars.append(rng.choice(unicode_atoms))
            else:
                chars.append(rng.choice(alphabet))
        text = "".join(chars)
        token_len = len(encode_unicode_text(text))
        if text not in seen and 1 <= token_len <= 16:
            seen.add(text)
            values.append(text)
    return values


def build_d512_copy_curriculum() -> D512CopyCurriculum:
    native = tuple(default_alphabet())
    train_texts = list(native)
    train_texts += ["CAT", "DOG", "A Z", "Aa0", "lambda λ", "中文", "🙂", "🧠", "\t", "\x00"]
    forbidden = set(train_texts)
    train_texts += _unique_random_texts(seed=2026092401, count=900, forbidden=forbidden)
    forbidden = set(train_texts)
    heldout_texts = _unique_random_texts(seed=2026092402, count=240, forbidden=forbidden)
    train = tuple(_case("train", text, index) for index, text in enumerate(train_texts))
    heldout = tuple(_case("heldout", text, 100_000 + index) for index, text in enumerate(heldout_texts))
    return D512CopyCurriculum(train_cases=train, heldout_cases=heldout)


def collate_d512_copy_cases(
    cases: Sequence[D512CopyCase],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not cases:
        raise ValueError("cannot collate an empty D512 copy batch")
    length = cases[0].transport_length
    if any(item.transport_length != length for item in cases):
        raise ValueError("D512 copy batches must be bucketed by exact transport length")
    token_ids = torch.tensor([item.token_ids for item in cases], dtype=torch.long, device=device)
    bank = unicode_transport_bank()
    arrays = [np.stack([bank[token_id] for token_id in item.token_ids], axis=0) for item in cases]
    cells = torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32, device=device)
    return cells, token_ids


def d512_copy_loss(
    model: ContinuousCoreD512,
    source_cells16: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> torch.Tensor:
    logits, _ = model.teacher_forced_logits(source_cells16, target_token_ids)
    eos = torch.full(
        (target_token_ids.shape[0], 1),
        model.eos_id,
        dtype=torch.long,
        device=target_token_ids.device,
    )
    targets = torch.cat((target_token_ids, eos), dim=1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def deterministic_copy_batches(
    curriculum: D512CopyCurriculum,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[tuple[D512CopyCase, ...], ...]:
    if batch_size < 1 or steps < 1:
        raise ValueError("batch_size and steps must be positive")
    buckets: dict[int, list[D512CopyCase]] = {}
    for case in curriculum.train_cases:
        buckets.setdefault(case.transport_length, []).append(case)
    lengths = tuple(sorted(length for length, cases in buckets.items() if len(cases) >= 1))
    rng = random.Random(seed)
    cursors = {length: 0 for length in lengths}
    for length in lengths:
        rng.shuffle(buckets[length])
    batches: list[tuple[D512CopyCase, ...]] = []
    for step in range(steps):
        length = lengths[step % len(lengths)]
        bucket = buckets[length]
        selected: list[D512CopyCase] = []
        for _ in range(batch_size):
            cursor = cursors[length]
            if cursor >= len(bucket):
                rng.shuffle(bucket)
                cursor = 0
            selected.append(bucket[cursor])
            cursors[length] = cursor + 1
        batches.append(tuple(selected))
    return tuple(batches)


@torch.no_grad()
def evaluate_d512_copy(
    model: ContinuousCoreD512,
    cases: Sequence[D512CopyCase],
    *,
    device: torch.device,
    max_cases: int = 120,
) -> D512CopyEvaluation:
    model.eval()
    selected = tuple(cases[:max_cases])
    if not selected:
        raise ValueError("evaluation requires heldout cases")
    content_correct = 0
    content_total = 0
    eos_correct = 0
    exact = 0
    terminated = 0
    valid_unicode = 0
    losses: list[float] = []
    label_counts = [0] * (model.cfg.output_classes)
    content_label_counts = [0] * model.cfg.transport_vocab_size

    for case in selected:
        cells, targets = collate_d512_copy_cases((case,), device=device)
        logits, _ = model.teacher_forced_logits(cells, targets)
        pred = logits[:, :-1].argmax(dim=-1)
        content_correct += int((pred == targets).sum().item())
        content_total += targets.numel()
        eos_correct += int(logits[:, -1].argmax(dim=-1).item() == model.eos_id)
        loss = d512_copy_loss(model, cells, targets)
        losses.append(float(loss.item()))
        for token_id in case.token_ids:
            label_counts[token_id] += 1
            content_label_counts[token_id] += 1
        label_counts[model.eos_id] += 1

        emitted, _, did_terminate = model.greedy_generate(
            cells,
            max_steps=max(24, len(case.token_ids) + 8),
        )
        terminated += int(did_terminate)
        exact += int(did_terminate and emitted == case.token_ids)
        try:
            decode_unicode_tokens(emitted)
        except Exception:
            pass
        else:
            valid_unicode += 1

    total_labels = sum(label_counts)
    constant_floor = max(label_counts) / total_labels
    content_constant_floor = max(content_label_counts) / content_total
    return D512CopyEvaluation(
        cases=len(selected),
        content_tokens=content_total,
        teacher_content_accuracy=content_correct / content_total,
        teacher_eos_accuracy=eos_correct / len(selected),
        free_exact_accuracy=exact / len(selected),
        termination_accuracy=terminated / len(selected),
        valid_unicode_accuracy=valid_unicode / len(selected),
        strongest_constant_floor=constant_floor,
        strongest_content_constant_floor=content_constant_floor,
        mean_loss=sum(losses) / len(losses),
    )


def d512_copy_objective_id() -> str:
    return canonical_sha256(
        {
            "schema": D512_COPY_OBJECTIVE_SCHEMA,
            "loss": "cross_entropy_transport_plus_private_eos",
            "transport_categories": TRANSPORT_VOCAB_SIZE,
            "eos_category": TRANSPORT_VOCAB_SIZE,
            "teacher_forcing": "predict-before-exact-d16-feedback",
        }
    )


def _write_preflight_payload(state_root: Path, kind: PreflightEvidenceKind, payload: dict[str, Any]) -> str:
    artifact_id = canonical_sha256(payload)
    directory = state_root / "training" / "trainer" / "preflight_evidence_d512"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact_id}.json"
    body = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError("D512 preflight evidence identity collision")
    else:
        path.write_bytes(body)
    return artifact_id


def build_d512_training_preflight(
    *,
    model: ContinuousCoreD512,
    curriculum: D512CopyCurriculum,
    inventory: ParameterInventory,
    plan: ParameterMutationPlanLike,
    batch_size: int,
    state_root: Path | str,
) -> TrainingPreflightReceipt:
    root = Path(state_root).resolve()
    descriptor = inventory.module(plan.module_id).descriptor
    if descriptor.d_model != 512:
        raise ValueError("D512 preflight requires a d_model=512 descriptor")

    train_tokens = {token for case in curriculum.train_cases for token in case.token_ids}
    native_covered = set(range(NATIVE_TOKEN_COUNT)).issubset(train_tokens)
    module_names = {type(module).__name__ for module in model.modules()}
    no_attention = not any("Attention" in name or "Transformer" in name for name in module_names)
    bank_exact = np.array_equal(model.bank16.detach().cpu().numpy(), unicode_transport_bank())

    probe_a = torch.tensor(np.stack([transport_token_cell16(token) for token in encode_unicode_text("CAT")]), dtype=torch.float32, device=model.bank16.device)
    probe_b = torch.tensor(np.stack([transport_token_cell16(token) for token in encode_unicode_text("DOG")]), dtype=torch.float32, device=model.bank16.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        state_a = model.ingest_cells(probe_a)
        state_b = model.ingest_cells(probe_b)
        counterfactual_delta = float((state_a - state_b).abs().max().item())
    model.train(was_training)

    clone = ContinuousCoreD512(model.cfg).to(model.bank16.device)
    clone.load_state_dict(model.state_dict(), strict=True)
    checkpoint_exact = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), clone.state_dict().values(), strict=True))

    payloads: dict[PreflightEvidenceKind, dict[str, Any]] = {
        PreflightEvidenceKind.STATIC_CAPACITY_SCAN: {
            "schema": "axon-d512-preflight-static-v1",
            "passed": no_attention,
            "module_types": sorted(module_names),
            "attention_modules": [],
            "sequence_ceiling_in_model_config": None,
        },
        PreflightEvidenceKind.ARCHITECTURE_CAPACITY: {
            "schema": "axon-d512-preflight-architecture-v1",
            "passed": bool(bank_exact and model.cfg.d_model == 512 and model.cfg.output_classes == TRANSPORT_VOCAB_SIZE + 1),
            "config": model.cfg.to_canonical_dict(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "bank_shape": list(model.bank16.shape),
            "bank_exact": bank_exact,
        },
        PreflightEvidenceKind.CURRICULUM_DISTRIBUTION: {
            "schema": "axon-d512-preflight-curriculum-v1",
            "passed": bool(native_covered and {item.text for item in curriculum.train_cases}.isdisjoint({item.text for item in curriculum.heldout_cases})),
            "train_cases": len(curriculum.train_cases),
            "heldout_cases": len(curriculum.heldout_cases),
            "train_min_tokens": min(item.transport_length for item in curriculum.train_cases),
            "train_max_tokens": max(item.transport_length for item in curriculum.train_cases),
            "heldout_min_tokens": min(item.transport_length for item in curriculum.heldout_cases),
            "heldout_max_tokens": max(item.transport_length for item in curriculum.heldout_cases),
            "native_coverage": len(set(range(NATIVE_TOKEN_COUNT)) & train_tokens),
            "native_required": NATIVE_TOKEN_COUNT,
            "split_disjoint": True,
        },
        PreflightEvidenceKind.BOUNDARY_COVERAGE: {
            "schema": "axon-d512-preflight-boundary-v1",
            "passed": True,
            "source": "complete exact D16 SharedField view per authored case",
            "target": "variable-length transport sequence plus private EOS",
            "observed_transport_lengths": sorted({item.transport_length for item in curriculum.train_cases + curriculum.heldout_cases}),
            "fixed_output_length": None,
        },
        PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE: {
            "schema": "axon-d512-preflight-counterfactual-v1",
            "passed": counterfactual_delta > 0.0,
            "probe_pair": ["CAT", "DOG"],
            "max_state_delta": counterfactual_delta,
            "claim": "input path is causally active; this is not a learned capability claim",
        },
        PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY: {
            "schema": "axon-d512-preflight-checkpoint-v1",
            "passed": checkpoint_exact,
            "strict_same_config_restore": checkpoint_exact,
            "config_id": model.cfg.config_id,
            "state_keys": sorted(model.state_dict()),
        },
    }
    failures = [kind.value for kind, payload in payloads.items() if not payload["passed"]]
    if failures:
        raise RuntimeError("D512 training preflight failed: " + ", ".join(failures))

    evidence: list[TrainingPreflightEvidence] = []
    for kind, payload in payloads.items():
        artifact_id = _write_preflight_payload(root, kind, payload)
        evidence.append(
            TrainingPreflightEvidence(
                kind=kind,
                artifact_id=artifact_id,
                summary={
                    PreflightEvidenceKind.STATIC_CAPACITY_SCAN: "No attention/Transformer module or model sequence ceiling in B4 baseline",
                    PreflightEvidenceKind.ARCHITECTURE_CAPACITY: "Exact Axon 351xD16 bank, D512 recurrent chamber, and 352-class transport+EOS output verified",
                    PreflightEvidenceKind.CURRICULUM_DISTRIBUTION: "Disjoint variable-length train/heldout curriculum covers all 95 native symbols",
                    PreflightEvidenceKind.BOUNDARY_COVERAGE: "Complete D16 fields feed variable-length exact transport targets with explicit private EOS",
                    PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE: "Changed exact D16 input changes the fresh recurrent state",
                    PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY: "Strict same-config state restoration reproduces every tensor and buffer",
                }[kind],
                passed=True,
            )
        )

    contract = CompleteFieldTrainingContract(
        module_id=descriptor.module_id,
        organ_kind=descriptor.organ_kind.value,
        architecture=descriptor.architecture,
        architecture_config_id=model.cfg.config_id,
        compiler_schema_ids=(D16_VIEW_SCHEMA, UNICODE_TRANSPORT_SCHEMA),
        source_position_scheme="canonical-d16-view-region-order-v1",
        target_position_scheme="sequential-transport-token-plus-private-eos-v1",
        declared_bounds=(
            DeclaredTrainingBound(
                name="smoke_batch_size",
                category=TrainingBoundCategory.ADMISSION_BUDGET,
                value=batch_size,
                source_preserved=True,
                continuation_or_failure="reduce batch size without changing model/curriculum identity",
            ),
            DeclaredTrainingBound(
                name="evaluation_generation_guard",
                category=TrainingBoundCategory.RESULT_COUNT_POLICY,
                value=64,
                source_preserved=True,
                continuation_or_failure="mark nontermination and preserve exact source evidence",
            ),
        ),
    )
    return build_training_preflight_receipt(
        contract=contract,
        inventory=inventory,
        plan=plan,
        evidence=evidence,
    )


__all__ = [
    "D512_COPY_CURRICULUM_SCHEMA",
    "D512_COPY_OBJECTIVE_SCHEMA",
    "D512CopyCase",
    "D512CopyCurriculum",
    "D512CopyEvaluation",
    "build_d512_copy_curriculum",
    "build_d512_training_preflight",
    "collate_d512_copy_cases",
    "d512_copy_loss",
    "d512_copy_objective_id",
    "deterministic_copy_batches",
    "evaluate_d512_copy",
]
