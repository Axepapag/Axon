"""Complete-Field launch firewall for learned Heart translation tissue."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from runtime.field import D64_COMPILER_SCHEMA, canonical_sha256
from runtime.heart.intelligence import CRITICAL_SEMANTIC_CLASSES
from runtime.heart.translation_core import HeartTranslationCore, heart_translation_architecture_id
from runtime.trainer import (
    CompleteFieldTrainingContract,
    DeclaredTrainingBound,
    OrganKind,
    ParameterInventory,
    ParameterMutationPlan,
    PreflightEvidenceKind,
    TrainingBoundCategory,
    TrainingPreflightEvidence,
    TrainingPreflightReceipt,
    build_training_preflight_receipt,
)

from .heart_translation import HeartTranslationCurriculum


HEART_PREFLIGHT_EVIDENCE_SCHEMA = "axon-heart-complete-field-preflight-evidence-v1"
FROZEN_SUBSTRATE_SCHEMA = "axon-frozen-alphabet-substrate-16d-v1"
_ACTIVE_SCAN_ROOTS = ("runtime", "training", "curator", "Cortext", "scripts")
_FORBIDDEN_IDENTIFIER_COMPONENTS = (
    ("max", "source", "chars"),
    ("max", "input", "chars"),
    ("max", "target", "chars"),
    ("max", "output", "chars"),
    ("max", "pages"),
    ("active", "tail", "chars"),
)


class HeartTrainingPreflightError(RuntimeError):
    pass


def _forbidden_identifiers() -> tuple[str, ...]:
    return tuple("_".join(parts) for parts in _FORBIDDEN_IDENTIFIER_COMPONENTS)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.Tuple, ast.List)):
        return ",".join(_target_name(item) for item in node.elts)
    return ""


def _textish_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    return ""


def _is_authoritative_text_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in {
        "input_ids",
        "source_ids",
        "target_ids",
        "source_indices",
        "target_indices",
        "tokens",
        "source_tokens",
        "target_tokens",
    }:
        return True
    if any(lowered.endswith(suffix) for suffix in ("_sha256", "_id", "_ids", "_ref", "_refs", "_count")):
        return False
    return lowered in {"text", "content", "summary", "source", "target", "input"} or any(
        lowered.endswith(suffix)
        for suffix in ("_text", "_content", "_summary", "_source", "_target", "_input")
    )


def _bounded_text_slice(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
        return None
    upper = node.slice.upper
    name = _textish_name(node.value)
    if not _is_authoritative_text_name(name):
        return None
    if upper is None:
        return None
    return name, ast.unparse(upper)


class _CapacityPoisonVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []

    def flag(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"{self.path}:{getattr(node, 'lineno', '?')}: {message}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.lower() in _forbidden_identifiers():
            self.flag(node, f"forbidden checkpoint-capacity identifier {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.lower() in _forbidden_identifiers():
            self.flag(node, f"forbidden checkpoint-capacity attribute {node.attr!r}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call) and _call_name(node.value.func).endswith("Embedding"):
            targets = " ".join(_target_name(target).lower() for target in node.targets)
            if "position" in targets or "page_embedding" in targets:
                self.flag(node, "finite learned position/page embedding")
        if isinstance(node.value, ast.Call) and _call_name(node.value.func).endswith("Parameter"):
            targets = " ".join(_target_name(target).lower() for target in node.targets)
            if "position" in targets or "page" in targets:
                self.flag(node, "finite learned position/page parameter")
        bounded_slice = _bounded_text_slice(node.value)
        if bounded_slice is not None:
            source_name, _ = bounded_slice
            target_names = {_target_name(target).lower() for target in node.targets}
            if source_name in target_names or any(_is_authoritative_text_name(name) for name in target_names):
                self.flag(node, "bounded destructive slice over authoritative text")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.value, ast.Call) and _call_name(node.value.func).endswith("Embedding"):
            target = _target_name(node.target).lower()
            if "position" in target or "page_embedding" in target:
                self.flag(node, "finite learned position/page embedding")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod):
            rendered = ast.unparse(node).lower()
            if "page" in rendered or "position" in rendered:
                self.flag(node, "page/position modulo alias")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _call_name(node.func)
        for keyword in node.keywords:
            if keyword.arg == "truncation" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self.flag(node, "tokenizer/collator truncation enabled")
            if call.endswith("load_state_dict") and keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                self.flag(node, "forgiving checkpoint load_state_dict(strict=False)")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and _bounded_text_slice(node.value) is not None:
            self.flag(node, "bounded returned slice over authoritative text")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = (node.left, *node.comparators)
        for left, right in zip(operands, operands[1:]):
            for length_node, constant_node in ((left, right), (right, left)):
                if not (
                    isinstance(length_node, ast.Call)
                    and _call_name(length_node.func) == "len"
                    and len(length_node.args) == 1
                ):
                    continue
                subject = _textish_name(length_node.args[0])
                is_fixed_number = (
                    isinstance(constant_node, ast.Constant)
                    and isinstance(constant_node.value, int)
                    and constant_node.value > 1
                )
                boundary_name = _textish_name(constant_node)
                is_named_capacity = any(
                    part in boundary_name for part in ("max", "limit", "window", "capacity", "cap")
                )
                if _is_authoritative_text_name(subject) and (is_fixed_number or is_named_capacity):
                    self.flag(node, "fixed numeric length guard over authoritative source/target text")
                    break
        self.generic_visit(node)


def scan_active_capacity_poison(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    files: list[Path] = []
    for name in _ACTIVE_SCAN_ROOTS:
        scan_root = root / name
        if not scan_root.exists():
            continue
        files.extend(
            path
            for path in scan_root.rglob("*")
            if path.is_file()
            and "archive" not in {part.lower() for part in path.parts}
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".json", ".toml", ".yaml", ".yml"}
        )

    violations: list[str] = []
    hashes: dict[str, str] = {}
    forbidden = _forbidden_identifiers()
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        hashes[relative] = _sha256_path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError as exc:
                violations.append(f"{relative}:{exc.lineno}: source cannot be parsed")
                continue
            visitor = _CapacityPoisonVisitor(Path(relative))
            visitor.visit(tree)
            violations.extend(visitor.violations)
        else:
            lowered = text.lower()
            for identifier in forbidden:
                if identifier in lowered:
                    violations.append(f"{relative}: forbidden capacity key {identifier!r}")
            compact = "".join(lowered.split())
            if '"truncation":true' in compact or "truncation=true" in compact:
                violations.append(f"{relative}: tokenizer/collator truncation enabled")

    return {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.STATIC_CAPACITY_SCAN.value,
        "roots": list(_ACTIVE_SCAN_ROOTS),
        "file_count": len(hashes),
        "source_sha256": hashes,
        "violations": sorted(violations),
        "passed": not violations,
    }


def _architecture_capacity(model: HeartTranslationCore) -> dict[str, Any]:
    learned_position_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding) and ("position" in name.lower() or "page" in name.lower())
    ]
    config_names = {name.lower() for name in model.cfg.__dataclass_fields__}
    forbidden_config_names = sorted(config_names & set(_forbidden_identifiers()))
    payload = {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.ARCHITECTURE_CAPACITY.value,
        "architecture": heart_translation_architecture_id(model.cfg),
        "config": model.cfg.to_canonical_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "learned_position_modules": learned_position_modules,
        "forbidden_config_names": forbidden_config_names,
        "source_position_scheme": "dynamic-sinusoidal-absolute-v1",
        "target_position_scheme": "dynamic-sinusoidal-absolute-v1",
        "ordered_source_sweeps": 2,
    }
    payload["passed"] = not learned_position_modules and not forbidden_config_names
    return payload


def _percentile(values: Iterable[int], fraction: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _split_distribution(cases, page_chars: int) -> dict[str, Any]:
    source_lengths = [len(case.source_text) for case in cases]
    target_lengths = [len(case.target_text) for case in cases]
    return {
        "cases": len(cases),
        "source_chars": {
            "min": min(source_lengths),
            "median": _percentile(source_lengths, 0.5),
            "p95": _percentile(source_lengths, 0.95),
            "max": max(source_lengths),
        },
        "target_chars": {
            "min": min(target_lengths),
            "median": _percentile(target_lengths, 0.5),
            "p95": _percentile(target_lengths, 0.95),
            "max": max(target_lengths),
        },
        "source_pages": {
            "max": max(math.ceil(length / page_chars) for length in source_lengths),
            "multi_page_cases": sum(length > page_chars for length in source_lengths),
        },
    }


def _curriculum_distribution(curriculum: HeartTranslationCurriculum, page_chars: int) -> dict[str, Any]:
    split_cases = {
        "train": curriculum.train_cases,
        "heldout": curriculum.heldout_cases,
        "regression": curriculum.regression_cases,
    }
    distributions = {name: _split_distribution(cases, page_chars) for name, cases in split_cases.items()}
    beyond: dict[str, dict[str, int]] = {}
    for split in ("train", "heldout"):
        cases = split_cases[split]
        beyond[split] = {
            semantic_class: sum(
                semantic_class in case.critical_classes
                and case.grounding_start >= page_chars
                and len(case.source_text) > page_chars
                for case in cases
            )
            for semantic_class in CRITICAL_SEMANTIC_CLASSES
        }
    passed = all(count > 0 for rows in beyond.values() for count in rows.values())
    return {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.CURRICULUM_DISTRIBUTION.value,
        "curriculum_id": curriculum.curriculum_id,
        "train_manifest_id": curriculum.train_manifest_id,
        "heldout_manifest_id": curriculum.heldout_manifest_id,
        "page_chars": page_chars,
        "distributions": distributions,
        "beyond_page_grounding_by_critical_class": beyond,
        "counterfactual_classes": sorted(pair.semantic_class for pair in curriculum.counterfactual_pairs),
        "passed": passed,
    }


def _source_batch(model: HeartTranslationCore, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(text) for text in texts)
    indices = torch.full((len(texts), width), model.source_pad_index, dtype=torch.long, device=model.device)
    mask = torch.zeros((len(texts), width), dtype=torch.bool, device=model.device)
    for row, text in enumerate(texts):
        encoded = torch.tensor([model.char_to_index[char] for char in text], dtype=torch.long, device=model.device)
        indices[row, : encoded.numel()] = encoded
        mask[row, : encoded.numel()] = True
    return indices, mask


@torch.no_grad()
def _boundary_coverage(model: HeartTranslationCore) -> dict[str, Any]:
    page = model.cfg.source_page_chars
    lengths = sorted(set((1, page - 1 if page > 1 else 1, page, page + 1, page * 2 + 1)))
    texts = ["a" * length for length in lengths]
    source_indices, source_mask = _source_batch(model, texts)
    dialects = torch.zeros(len(texts), dtype=torch.long, device=model.device)
    destinations = torch.ones(len(texts), dtype=torch.long, device=model.device)
    query_states, memory, coverage = model._encode(source_indices, source_mask, dialects, destinations)
    expected_hashes = tuple(
        hashlib.sha256(bytes([model.char_to_index["a"]]) * length).hexdigest()
        for length in lengths
    )
    passed = bool(
        coverage.complete
        and coverage.sweeps == 2
        and coverage.source_characters == tuple(lengths)
        and coverage.source_index_sha256 == expected_hashes
        and coverage.visited_characters_per_sweep == (tuple(lengths), tuple(lengths))
        and memory.shape[:2] == source_indices.shape
        and query_states.shape == (len(texts), len(model.query_names), model.cfg.d_model)
    )
    return {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.BOUNDARY_COVERAGE.value,
        "page_chars": page,
        "tested_source_lengths": lengths,
        "source_index_sha256": list(coverage.source_index_sha256),
        "visited_characters_per_sweep": [list(row) for row in coverage.visited_characters_per_sweep],
        "memory_shape": list(memory.shape),
        "query_shape": list(query_states.shape),
        "passed": passed,
    }


@torch.no_grad()
def _counterfactual_dependence(model: HeartTranslationCore) -> dict[str, Any]:
    length = model.cfg.source_page_chars * 2 + 1
    positions = (0, length // 2, length - 1)
    base = "a" * length
    texts = [base]
    for position in positions:
        changed = list(base)
        changed[position] = "b"
        texts.append("".join(changed))
    source_indices, source_mask = _source_batch(model, texts)
    source_dialects = torch.zeros(len(texts), dtype=torch.long, device=model.device)
    destination_dialects = torch.ones(len(texts), dtype=torch.long, device=model.device)
    decoder_input = torch.full((len(texts), 1), model.bos_index, dtype=torch.long, device=model.device)
    output = model(source_indices, source_mask, source_dialects, destination_dialects, decoder_input)
    observable = torch.cat(
        (
            output.target_log_probs[:, 0],
            *(output.semantic_logits[name] for name in sorted(output.semantic_logits)),
            output.referent_start_logits,
            output.referent_end_logits,
            output.grounding_start_logits,
            output.grounding_end_logits,
        ),
        dim=1,
    )
    deltas = [float((observable[row] - observable[0]).abs().max().cpu()) for row in range(1, len(texts))]
    hash_changed = [
        output.source_coverage.source_index_sha256[row] != output.source_coverage.source_index_sha256[0]
        for row in range(1, len(texts))
    ]
    passed = all(value > 0.0 for value in deltas) and all(hash_changed) and output.source_coverage.complete
    return {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE.value,
        "source_length": length,
        "changed_positions": list(positions),
        "observable_max_abs_deltas": deltas,
        "source_identity_changed": hash_changed,
        "note": "Structural input-use proof at untrained initialization; not a semantic capability claim.",
        "passed": passed,
    }


def _checkpoint_compatibility(model: HeartTranslationCore) -> dict[str, Any]:
    state = model.state_dict()
    exact = HeartTranslationCore(model.cfg)
    exact.load_state_dict(state, strict=True)
    incompatible_cfg = replace(model.cfg, d_model=model.cfg.d_model * 2)
    if incompatible_cfg.d_model % incompatible_cfg.n_heads:
        incompatible_cfg = replace(incompatible_cfg, n_heads=1)
    incompatible = HeartTranslationCore(incompatible_cfg)
    incompatible_rejected = False
    try:
        incompatible.load_state_dict(state, strict=True)
    except RuntimeError:
        incompatible_rejected = True
    return {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "check": PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY.value,
        "exact_config_load": True,
        "incompatible_config": incompatible_cfg.to_canonical_dict(),
        "incompatible_strict_load_rejected": incompatible_rejected,
        "state_key_count": len(state),
        "passed": incompatible_rejected,
    }


def _write_immutable_evidence(state_root: Path, kind: PreflightEvidenceKind, payload: dict[str, Any]) -> str:
    artifact_id = canonical_sha256(payload)
    root = state_root / "training" / "heart" / "preflight_evidence"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    wrapped = {
        "schema": HEART_PREFLIGHT_EVIDENCE_SCHEMA,
        "artifact_id": artifact_id,
        "kind": kind.value,
        "payload": payload,
    }
    serialized = json.dumps(wrapped, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise HeartTrainingPreflightError(f"immutable preflight evidence disagrees at {path}")
        return artifact_id
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact_id


def build_heart_training_preflight(
    *,
    model: HeartTranslationCore,
    curriculum: HeartTranslationCurriculum,
    inventory: ParameterInventory,
    plan: ParameterMutationPlan,
    batch_size: int,
    state_root: Path | str,
    repo_root: Path | str,
) -> TrainingPreflightReceipt:
    if inventory.module(plan.module_id).descriptor.organ_kind is not OrganKind.HEART_TRANSLATION_CORE:
        raise HeartTrainingPreflightError("Heart preflight requires a Heart translation module")
    if batch_size < 1:
        raise HeartTrainingPreflightError("batch_size must be positive")

    was_training = model.training
    model.eval()
    try:
        payloads = {
            PreflightEvidenceKind.STATIC_CAPACITY_SCAN: scan_active_capacity_poison(repo_root),
            PreflightEvidenceKind.ARCHITECTURE_CAPACITY: _architecture_capacity(model),
            PreflightEvidenceKind.CURRICULUM_DISTRIBUTION: _curriculum_distribution(
                curriculum, model.cfg.source_page_chars
            ),
            PreflightEvidenceKind.BOUNDARY_COVERAGE: _boundary_coverage(model),
            PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE: _counterfactual_dependence(model),
            PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY: _checkpoint_compatibility(model),
        }
    finally:
        model.train(was_training)

    failures = [kind.value for kind, payload in payloads.items() if not payload.get("passed")]
    if failures:
        details = {
            kind.value: payload.get("violations", payload)
            for kind, payload in payloads.items()
            if not payload.get("passed")
        }
        raise HeartTrainingPreflightError(
            "Heart training preflight failed: " + ", ".join(failures) + " :: " + json.dumps(details, sort_keys=True)
        )

    state = Path(state_root).resolve(strict=False)
    evidence: list[TrainingPreflightEvidence] = []
    for kind, payload in payloads.items():
        artifact_id = _write_immutable_evidence(state, kind, payload)
        evidence.append(
            TrainingPreflightEvidence(
                kind=kind,
                artifact_id=artifact_id,
                summary={
                    PreflightEvidenceKind.STATIC_CAPACITY_SCAN: "Active Python/config capacity scan passed",
                    PreflightEvidenceKind.ARCHITECTURE_CAPACITY: "Heart architecture has dynamic positions and no declared character ceiling",
                    PreflightEvidenceKind.CURRICULUM_DISTRIBUTION: "Train and holdout cover every critical class beyond one physical page",
                    PreflightEvidenceKind.BOUNDARY_COVERAGE: "Exact two-sweep coverage passed across physical-page boundaries",
                    PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE: "Head, middle, and tail source changes alter observable model output",
                    PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY: "Exact config loads and incompatible anatomy fails strict restore",
                }[kind],
                passed=True,
            )
        )

    descriptor = inventory.module(plan.module_id).descriptor
    architecture_config_id = canonical_sha256(
        {"architecture": descriptor.architecture, "config": model.cfg.to_canonical_dict()}
    )
    contract = CompleteFieldTrainingContract(
        module_id=descriptor.module_id,
        organ_kind=descriptor.organ_kind.value,
        architecture=descriptor.architecture,
        architecture_config_id=architecture_config_id,
        compiler_schema_ids=(FROZEN_SUBSTRATE_SCHEMA, D64_COMPILER_SCHEMA),
        source_position_scheme="dynamic-sinusoidal-absolute-v1",
        target_position_scheme="dynamic-sinusoidal-absolute-v1",
        declared_bounds=(
            DeclaredTrainingBound(
                name="heart_source_page_chars",
                category=TrainingBoundCategory.PHYSICAL_PROCESSING_UNIT,
                value=model.cfg.source_page_chars,
                source_preserved=True,
                continuation_or_failure="Two ordered recurrent sweeps continue until every exact source character is visited.",
            ),
            DeclaredTrainingBound(
                name="training_batch_size",
                category=TrainingBoundCategory.COMPUTE_BUDGET,
                value=batch_size,
                source_preserved=True,
                continuation_or_failure="Deterministic batching selects whole curriculum cases and never slices a case.",
            ),
            DeclaredTrainingBound(
                name="optimizer_steps",
                category=TrainingBoundCategory.OPTIMIZATION_BUDGET,
                value=plan.max_steps,
                source_preserved=True,
                continuation_or_failure="The bounded candidate ends explicitly and cannot imply serving capability without promotion evidence.",
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
    "HEART_PREFLIGHT_EVIDENCE_SCHEMA",
    "FROZEN_SUBSTRATE_SCHEMA",
    "HeartTrainingPreflightError",
    "scan_active_capacity_poison",
    "build_heart_training_preflight",
]
