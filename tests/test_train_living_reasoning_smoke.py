from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.train_living_reasoning_smoke import _recover_resume_boundary


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
