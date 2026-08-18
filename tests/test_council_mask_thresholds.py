from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.council.engine import CouncilEngine, retention_offset


def test_retention_offsets_by_exact_unit() -> None:
    history = (
        "User: one\nAssistant: alpha\n"
        "User: two\nAssistant: beta\n"
        "User: three\nAssistant: gamma"
    )
    offset = retention_offset(history, "turns", 2, "conversation_history")
    assert history[offset:].startswith("User: two")
    assert retention_offset(history, "turns", 50, "conversation_history") == 0
    assert retention_offset(history, "turns", 0, "conversation_history") == len(history)
    assert retention_offset("a\nb\nc", "lines", 2, "scratch") == 2
    assert retention_offset("one\n\ntwo\n\nthree", "paragraphs", 2, "diary") == 5
    with pytest.raises(ValueError, match="only valid"):
        retention_offset(history, "turns", 2, "scratch")


def test_engine_persists_threshold_policy(tmp_path: Path) -> None:
    field_path = tmp_path / "field.json"
    config_path = tmp_path / "config.json"
    config = CouncilEngine.default_config()
    config.update({
        "field_state_path": str(field_path),
        "dormant_tails_path": str(tmp_path / "tails.jsonl"),
        "souls_path": str(tmp_path / "souls"),
    })
    config_path.write_text(json.dumps(config), encoding="utf-8")
    engine = CouncilEngine(config_path, lambda event: None)
    history = (
        "User: one\nAssistant: alpha\n"
        "User: two\nAssistant: beta\n"
        "User: three\nAssistant: gamma"
    )
    engine.set_region("conversation_history", history)
    view = engine.set_mask("conversation_history", unit="turns", retain=2)
    assert view["active"].startswith("User: two")
    assert view["mask_mode"] == "threshold"
    assert view["mask_unit"] == "turns"
    assert view["mask_retain"] == 2

    reloaded = CouncilEngine(config_path, lambda event: None)
    persisted = reloaded.field_view()["regions"]["conversation_history"]
    assert persisted["active"].startswith("User: two")
    assert persisted["mask_unit"] == "turns"
    assert persisted["mask_retain"] == 2
