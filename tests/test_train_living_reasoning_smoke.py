from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime.trainer import ResourceTranche, TrancheStore
from scripts.train_living_reasoning_smoke import (
    _persist_tranche_lineage,
    _recover_resume_boundary,
)


class _FakeBundles:
    def __init__(self, *, recovered=(), latest=None) -> None:
        self.recovered = tuple(recovered)
        self.latest = latest
        self.calls: list[tuple[str, str, str]] = []

    def recover_pending(self, module_id: str, candidate_generation: str):
        self.calls.append(("recover", module_id, candidate_generation))
        return self.recovered

    def latest_bundle(self, module_id: str, candidate_generation: str):
        self.calls.append(("latest", module_id, candidate_generation))
        return self.latest


def test_resume_boundary_recovers_before_reading_latest() -> None:
    recovered = (SimpleNamespace(bundle_id="bundle-1", step=1),)
    latest = SimpleNamespace(bundle_id="bundle-1", step=1)
    bundles = _FakeBundles(recovered=recovered, latest=latest)

    observed_recovered, observed_latest = _recover_resume_boundary(
        bundles,
        "core-a",
        "candidate-a",
        resume=True,
    )

    assert observed_recovered == recovered
    assert observed_latest is latest
    assert [call[0] for call in bundles.calls] == ["recover", "latest"]


def test_resume_boundary_requires_resume_for_existing_work() -> None:
    bundles = _FakeBundles(latest=SimpleNamespace(bundle_id="bundle-1", step=1))
    with pytest.raises(RuntimeError, match="pass --resume"):
        _recover_resume_boundary(bundles, "core-a", "candidate-a", resume=False)
    assert [call[0] for call in bundles.calls] == ["recover", "latest"]


def test_resume_boundary_fails_when_resume_has_no_recovered_or_accepted_work() -> None:
    bundles = _FakeBundles()
    with pytest.raises(RuntimeError, match="no accepted English candidate bundle"):
        _recover_resume_boundary(bundles, "core-a", "candidate-a", resume=True)
    assert [call[0] for call in bundles.calls] == ["recover", "latest"]


def test_persist_tranche_lineage_writes_fresh_tranche_without_continuation(tmp_path) -> None:
    store = TrancheStore(tmp_path / "training" / "trainer")
    tranche = ResourceTranche(
        module_id="core-a",
        candidate_generation_id="candidate-a",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=0,
        steps=2,
    )

    tranche_path, continuation_path, continuation = _persist_tranche_lineage(
        store,
        tranche,
        None,
    )

    assert tranche_path == store.tranches_dir / f"{tranche.tranche_id}.json"
    assert tranche_path.is_file()
    assert continuation_path is None
    assert continuation is None
    assert store.read_tranche(tranche.tranche_id) == tranche
    assert store.continuations_for("core-a", "candidate-a") == ()


def test_persist_tranche_lineage_binds_resume_to_exact_parent(tmp_path) -> None:
    store = TrancheStore(tmp_path / "training" / "trainer")
    initial = ResourceTranche(
        module_id="core-a",
        candidate_generation_id="candidate-a",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=0,
        steps=2,
    )
    _persist_tranche_lineage(store, initial, None)
    latest = SimpleNamespace(
        step=2,
        bundle_id="c" * 64,
        checkpoint_id="d" * 64,
        optimization_receipt_id="e" * 64,
        after_soul_id="f" * 64,
    )
    continuation_tranche = ResourceTranche(
        module_id="core-a",
        candidate_generation_id="candidate-a",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=2,
        steps=2,
        parent_bundle_id=latest.bundle_id,
    )

    tranche_path, continuation_path, continuation = _persist_tranche_lineage(
        store,
        continuation_tranche,
        latest,
    )

    assert tranche_path.is_file()
    assert continuation is not None
    assert continuation.parent_bundle_id == latest.bundle_id
    assert continuation.parent_checkpoint_id == latest.checkpoint_id
    assert continuation.parent_optimizer_receipt_id == latest.optimization_receipt_id
    assert continuation.parent_soul_id == latest.after_soul_id
    assert continuation.prior_tranche_id == initial.tranche_id
    assert continuation_path == store.continuations_dir / f"{continuation.continuation_id}.json"
    assert store.read_continuation(continuation.continuation_id) == continuation
