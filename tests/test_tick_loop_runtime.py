from __future__ import annotations

import os
import time
from pathlib import Path

from runtime.tick_loop import parse_checkpoint_overrides, resolve_checkpoint, response_stats


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    now = time.time()
    os.utime(path, (now, now))


def test_resolve_checkpoint_prefers_story_and_ignores_part_files(tmp_path: Path) -> None:
    phase0 = tmp_path / "runs" / "mirror" / "coreA_charslot_phase0" / "ckpt_9.pt"
    story_part = tmp_path / "runs" / "mirror" / "coreA_charslot_stories" / "ckpt_10.pt.part"
    story = tmp_path / "runs" / "mirror" / "coreA_charslot_stories" / "ckpt_1.pt"
    _touch(phase0)
    _touch(story_part)
    _touch(story)

    assert resolve_checkpoint(tmp_path, "coreA") == story


def test_checkpoint_override_can_be_relative(tmp_path: Path) -> None:
    ckpt = tmp_path / "custom" / "core.pt"
    _touch(ckpt)
    overrides = parse_checkpoint_overrides(["coreA=custom/core.pt"])
    assert resolve_checkpoint(tmp_path, "coreA", overrides) == ckpt


def test_response_stats_reports_degeneracy() -> None:
    assert response_stats("") == "chars=0 uniq=0 top=0.000"
    assert response_stats('""""----') == "chars=8 uniq=2 top=0.500"
