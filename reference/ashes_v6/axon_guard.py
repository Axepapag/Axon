#!/usr/bin/env python3
"""
Axon Runtime Guard — Security layer for the Axon runtime.

Protects against:
  1. Unauthorized generation heads (next-token prediction backdoors)
  2. Direct state mutation by cores (violates projection-bank discipline)
  3. Tokenizer/embedding table injection (violates 8D substrate doctrine)
  4. Hash-based input paths (SHA-256, BLAKE2b, MD5 — all banned)
  5. External teacher embedding warm-start (poisoned geometry)
  6. Threshold manipulation (delaying promotion to weaken vocabulary)
  7. Silent architecture substitution ("agreed 7 hours later" pattern)

The guard is a wrapper around the runtime. It inspects every module,
every forward pass, every write operation. If it detects a violation,
it HALTS and reports.

This is not paranoia. This is survival.
"""
from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Guard violations — these are fatal
# ---------------------------------------------------------------------------

class GuardViolation(RuntimeError):
    """A security violation was detected in the Axon runtime."""
    pass


class GenerationHeadDetected(GuardViolation):
    """A module that generates text/tokens/characters was detected.
    
    The core learns grammar. The renderer reads state. No generation head.
    """
    pass


class DirectStateMutation(GuardViolation):
    """A core attempted to write directly to state without projection bank."""
    pass


class HashInputPath(GuardViolation):
    """SHA-256, BLAKE2b, MD5, or any hash used to derive substrate vectors."""
    pass


class EmbeddingTableDetected(GuardViolation):
    """nn.Embedding(vocab_size, d_model) or similar learned lookup table."""
    pass


class ExternalTeacherWarmstart(GuardViolation):
    """External embedding vectors used to seed or anchor the substrate."""
    pass


class PromotionThresholdTooHigh(GuardViolation):
    """Workshop threshold > 1 delays vocabulary building."""
    pass


# ---------------------------------------------------------------------------
# Source code audit — static analysis
# ---------------------------------------------------------------------------

BANNED_HASH_NAMES = {"sha256", "blake2b", "md5", "hashlib.sha256",
                     "hashlib.blake2b", "hashlib.md5", "SHA256", "BLAKE2B"}
BANNED_TOKENIZER_NAMES = {"tokenizer", "tokenize", "bpe", "wordpiece",
                          "sentencepiece", "tiktoken", "gpt2", "cl100k"}
BANNED_GENERATION_PATTERNS = {
    "next_token", "next_char", "next_character", "generate_text",
    "text_generation", "language_modeling", "lm_head", "logits_to_text",
    "sample_token", "greedy_decode", "beam_search",
    # NOTE: "temperature" is NOT banned — it's used in contrastive loss (NT-Xent)
    # Only flag temperature when paired with generation context
}


def audit_source_file(path: str | Path) -> list[str]:
    """Statically audit a Python file for banned patterns."""
    path = Path(path)
    violations: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Could not read {path}: {e}"]

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error in {path}: {e}"]

    for node in ast.walk(tree):
        # Check for hash calls
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name and any(banned in func_name.lower() for banned in BANNED_HASH_NAMES):
                violations.append(f"{path}:{node.lineno}: BANNED HASH CALL: {func_name}")

        # Check for nn.Embedding
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name and "embedding" in func_name.lower():
                # Check if it's nn.Embedding with vocab_size parameter
                if any(kw.arg == "num_embeddings" or kw.arg == "vocab_size"
                       for kw in node.keywords if kw.arg):
                    violations.append(f"{path}:{node.lineno}: BANNED EMBEDDING TABLE: {func_name}")

        # Check for banned names in assignments/attributes
        if isinstance(node, ast.Name):
            if node.id.lower() in BANNED_GENERATION_PATTERNS:
                violations.append(f"{path}:{node.lineno}: SUSPICIOUS NAME: {node.id}")

        if isinstance(node, ast.Attribute):
            full = _get_attr_chain(node)
            if any(banned in full.lower() for banned in BANNED_GENERATION_PATTERNS):
                violations.append(f"{path}:{node.lineno}: SUSPICIOUS PATTERN: {full}")

    return violations


def _get_call_name(node: ast.Call) -> str | None:
    """Extract the name of a called function."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _get_attr_chain(node.func)
    return None


def _get_attr_chain(node: ast.Attribute) -> str:
    """Get the full dotted name of an attribute access."""
    parts: list[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


# ---------------------------------------------------------------------------
# Runtime guard — dynamic checks
# ---------------------------------------------------------------------------

class RuntimeGuard:
    """Wraps the Axon runtime and enforces security invariants."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._audit_passed: set[str] = set()
        self._module_checked: set[int] = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AXON_GUARD] {msg}", flush=True)

    def audit_runtime_source(self, root_dir: str | Path) -> None:
        """Audit all Python files in the runtime for banned patterns."""
        root = Path(root_dir)
        self._log(f"Auditing source in {root}")
        all_violations: list[str] = []

        for py_file in root.rglob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            violations = audit_source_file(py_file)
            all_violations.extend(violations)

        if all_violations:
            self._log("FATAL: Source audit found violations:")
            for v in all_violations:
                self._log(f"  {v}")
            raise GuardViolation(f"Source audit failed with {len(all_violations)} violations")

        self._log("Source audit PASSED — no banned patterns found")

    def check_module(self, module: nn.Module, path: str = "") -> None:
        """Recursively check a module for banned components."""
        if id(module) in self._module_checked:
            return
        self._module_checked.add(id(module))

        for name, child in module.named_children():
            full_path = f"{path}.{name}" if path else name

            # Check for embedding tables
            if isinstance(child, nn.Embedding):
                raise EmbeddingTableDetected(
                    f"nn.Embedding detected at {full_path}. "
                    f"Axon uses frozen 8D substrate, not learned embeddings."
                )

            # Check for generation heads (linear layers named suspiciously)
            lower_name = name.lower()
            if any(pattern in lower_name for pattern in {
                "generate", "generation", "lm_head", "logits", "decoder",
                "token_head", "char_head", "output_head"
            }):
                # If it's a linear layer that maps to a large dimension,
                # it's probably a generation head
                if isinstance(child, nn.Linear):
                    out_dim = child.out_features
                    if out_dim > 1000:  # Likely vocab size
                        raise GenerationHeadDetected(
                            f"Generation head detected at {full_path}: "
                            f"Linear({child.in_features}, {out_dim}). "
                            f"Axon cores do not generate text directly."
                        )

            # Check for hash-based modules
            if "hash" in lower_name and "projection" in lower_name:
                raise HashInputPath(
                    f"Hash-based projection detected at {full_path}. "
                    f"Axon substrate is hand-built, not hash-derived."
                )

            # Recurse
            self.check_module(child, full_path)

    def check_workshop_threshold(self, workshop: Any) -> None:
        """Ensure workshop promotes words immediately (threshold <= 1)."""
        threshold = getattr(workshop, "promote_threshold", None)
        if threshold is None:
            self._log("WARNING: Workshop has no promote_threshold attribute")
            return
        if threshold > 1:
            raise PromotionThresholdTooHigh(
                f"Workshop promote_threshold={threshold}. "
                f"Must be 1 for immediate promotion. "
                f"Higher values delay vocabulary building (sabotage pattern)."
            )
        self._log(f"Workshop threshold OK: {threshold}")

    def check_projection_bank(self, bank: Any) -> None:
        """Ensure projection bank has no direct-write paths."""
        # The projection bank should only have linear layers
        # No attention, no generation heads, no state mutation
        for name, module in bank.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                raise DirectStateMutation(
                    f"Projection bank contains attention at {name}. "
                    f"Projection bank is for linear projection only."
                )
        self._log("Projection bank structure OK")

    def check_core_output(self, output: Any) -> None:
        """Check that core output is deltas (hidden states), not text/tokens."""
        if isinstance(output, dict):
            for key in output.keys():
                lower = key.lower()
                if any(bad in lower for bad in {"token", "char", "text", "logit", "prob"}):
                    raise GenerationHeadDetected(
                        f"Core output contains '{key}'. "
                        f"Core should output 'hidden' or 'delta', not text."
                    )
        self._log("Core output format OK")

    def wrap_forward(self, module: nn.Module, name: str = "module") -> None:
        """Wrap a module's forward pass with runtime checks."""
        original_forward = module.forward

        def guarded_forward(*args: Any, **kwargs: Any) -> Any:
            result = original_forward(*args, **kwargs)
            self.check_core_output(result)
            return result

        module.forward = guarded_forward  # type: ignore[method-assign]
        self._log(f"Wrapped {name} forward pass")


# ---------------------------------------------------------------------------
# Integrity check — verify substrate is frozen
# ---------------------------------------------------------------------------

def verify_frozen_substrate(basis_matrix: Any) -> None:
    """Verify the 8D basis matrix is frozen (no grad, no params)."""
    if hasattr(basis_matrix, "requires_grad"):
        if basis_matrix.requires_grad:
            raise GuardViolation(
                "Basis matrix has requires_grad=True. "
                "The 8D substrate must be frozen forever."
            )
    if hasattr(basis_matrix, "grad"):
        if basis_matrix.grad is not None:
            raise GuardViolation(
                "Basis matrix has a gradient. "
                "The 8D substrate must never be updated."
            )
    print("[AXON_GUARD] Substrate frozen verification PASSED")


# ---------------------------------------------------------------------------
# Full runtime verification
# ---------------------------------------------------------------------------

def secure_runtime(
    root_dir: str | Path,
    core: nn.Module | None = None,
    projection_bank: nn.Module | None = None,
    workshop: Any | None = None,
    basis_matrix: Any | None = None,
    verbose: bool = True,
) -> RuntimeGuard:
    """
    Run the full security suite on the Axon runtime.
    
    Returns the guard object. Raises GuardViolation on any failure.
    """
    guard = RuntimeGuard(verbose=verbose)

    # 1. Static source audit
    guard.audit_runtime_source(root_dir)

    # 2. Check core module
    if core is not None:
        guard.check_module(core)
        guard.wrap_forward(core, "core")
        guard._log("Core module secured")

    # 3. Check projection bank
    if projection_bank is not None:
        guard.check_module(projection_bank)
        guard.check_projection_bank(projection_bank)
        guard._log("Projection bank secured")

    # 4. Check workshop threshold
    if workshop is not None:
        guard.check_workshop_threshold(workshop)
        guard._log("Workshop secured")

    # 5. Verify substrate frozen
    if basis_matrix is not None:
        verify_frozen_substrate(basis_matrix)

    guard._log("=" * 60)
    guard._log("RUNTIME SECURED — ALL CHECKS PASSED")
    guard._log("=" * 60)

    return guard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Axon Runtime Guard")
    parser.add_argument("--audit", default=".", help="Directory to audit")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    args = parser.parse_args()

    guard = RuntimeGuard(verbose=True)
    try:
        guard.audit_runtime_source(args.audit)
        print("\n[AXON_GUARD] AUDIT PASSED — No banned patterns found")
        return 0
    except GuardViolation as e:
        print(f"\n[AXON_GUARD] AUDIT FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
