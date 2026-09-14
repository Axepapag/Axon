"""The Kaggle D64 screen shards must cover the declared opening exactly once."""

from __future__ import annotations

import json
from pathlib import Path

from training import d64_architecture_screening_tournament


ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "configs" / "kaggle" / "d64_architecture_screen_stage1_shards"


def test_stage_one_shards_cover_every_candidate_with_bounded_checkpoints() -> None:
    expected = {candidate.label for candidate in d64_architecture_screening_tournament().candidates}
    paths = sorted(SHARDS.glob("*.json"))
    assert {path.stem for path in paths} == expected

    observed: list[str] = []
    for path in paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        argv = config["entrypoint_argv"]
        assert config["schema"] == "axon-cloud-training-job-config-v1"
        assert config["provider"] == "kaggle"
        assert config["accelerator"] == "gpu"
        assert config.get("sync_mid_run") is not True
        assert argv[argv.index("--profile") + 1] == "architecture-screen"
        assert argv[argv.index("--tranche-steps") + 1] == "32"
        assert argv[argv.index("--checkpoint-interval") + 1] == "16"
        label = argv[argv.index("--candidate-label") + 1]
        assert label == path.stem
        observed.append(label)

    assert len(observed) == len(set(observed)) == 16
