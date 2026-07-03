"""Gate 1: unit tests for the Round Table Orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.table import buslink, manifests, prompts, replies, transcript
from runtime.table.prompts import BudgetView, PromptContext


FIXTURES = Path(__file__).parent / "fixtures" / "table"


class TestManifests:
    def test_load_all_manifests(self):
        loaded = manifests.load_all_manifests()
        assert "echo" in loaded
        assert "kimi" in loaded
        assert "hermes" in loaded
        assert loaded["echo"].wake.prompt_via == "stdin"
        assert "{prompt}" in loaded["kimi"].wake.argv[2]

    def test_manifest_validation_missing_client_id(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"display_name": "x"}))
        with pytest.raises(ValueError, match="missing 'client_id'"):
            manifests.load_manifest(path)

    def test_manifest_no_tokens(self):
        loaded = manifests.load_all_manifests()
        for manifest in loaded.values():
            raw = json.dumps(manifests.load_manifest.__wrapped__ if hasattr(manifests.load_manifest, "__wrapped__") else {})
            # Re-read raw file to inspect for secrets.
            p = manifests.ManifestDir / f"{manifest.client_id}.json"
            text = p.read_text()
            assert "BUS_TOKEN" not in text
            assert "token" not in text.lower() or "auth_token" not in text


class TestReplies:
    def test_valid_post(self):
        stdout = 'some text\n```json\n{"type": "post", "text": "hello", "stamp": "A / m / 2026-07-03"}\n```'
        action = replies.parse_reply(stdout)
        assert action["type"] == "post"
        assert action["text"] == "hello"

    def test_actions_array(self):
        stdout = '```json\n[{"type": "post", "text": "a", "stamp": "s"}, {"type": "pass", "stamp": "s"}]\n```'
        actions = replies.parse_reply_all(stdout)
        assert len(actions) == 2
        assert actions[0]["type"] == "post"
        assert actions[1]["type"] == "pass"

    def test_last_fenced_block_wins(self):
        stdout = (
            '```json\n{"type": "post", "text": "first", "stamp": "s"}\n```\n'
            'more\n'
            '```json\n{"type": "post", "text": "last", "stamp": "s"}\n```'
        )
        action = replies.parse_reply(stdout)
        assert action["text"] == "last"

    def test_garbage_fallback(self):
        stdout = "this is just plain text with no json"
        action = replies.parse_reply(stdout)
        assert action["type"] == "post"
        assert action["parse_fallback"] is True
        assert action["text"] == stdout

    def test_empty_output_fallback(self):
        action = replies.parse_reply("")
        assert action["type"] == "post"
        assert action["parse_fallback"] is True
        assert action["text"] == ""


class TestPrompts:
    def test_preamble_loaded(self):
        preamble = prompts.load_preamble()
        assert "Working Contract" in preamble
        assert "HALT AND FLAG" in preamble

    def test_digest_within_budget(self):
        entries = [{"seat": f"echo-{i}", "text": "x" * 100} for i in range(5)]
        ctx = PromptContext(
            round_id="r",
            cycle=1,
            max_cycles=2,
            seat="echo",
            seat_display="Echo",
            standing_question="q",
            officers=["codex"],
            synthesizer="claude",
            budgets=BudgetView(max_invocations=10, board_view_events=3, board_view_chars=500),
            doctrine_stamp={"sot_path": "docs/SOURCE_OF_TRUTH.md", "sot_sha256": "abc"},
            brief_text="brief",
            transcript_entries=entries,
            unread_dms=[],
        )
        prompt, meta = prompts.build_prompt(ctx)
        assert meta["digest"]["shown_entries"] <= 3
        assert meta["digest"]["truncated_count"] == 2
        assert "digest truncated" in prompt

    def test_no_truncation_when_under_budget(self):
        entries = [{"seat": "echo", "text": "short"}]
        ctx = PromptContext(
            round_id="r",
            cycle=1,
            max_cycles=2,
            seat="echo",
            seat_display="Echo",
            standing_question="q",
            officers=[],
            synthesizer="claude",
            budgets=BudgetView(max_invocations=10, board_view_events=10, board_view_chars=10000),
            doctrine_stamp={"sot_path": "p", "sot_sha256": "abc"},
            brief_text="brief",
            transcript_entries=entries,
            unread_dms=[],
        )
        prompt, meta = prompts.build_prompt(ctx)
        assert meta["digest"]["truncated"] is False
        assert "digest truncated" not in prompt

    def test_prompt_has_no_bus_token(self):
        ctx = PromptContext(
            round_id="r",
            cycle=1,
            max_cycles=2,
            seat="echo",
            seat_display="Echo",
            standing_question="q",
            officers=[],
            synthesizer="claude",
            budgets=BudgetView(max_invocations=10, board_view_events=10, board_view_chars=10000),
            doctrine_stamp={"sot_path": "p", "sot_sha256": "abc"},
            brief_text="brief",
            transcript_entries=[],
            unread_dms=[],
        )
        prompt, _ = prompts.build_prompt(ctx)
        assert "BUS_TOKEN" not in prompt
        assert "auth_token" not in prompt


class TestTranscript:
    def test_append_only(self, tmp_path):
        d = tmp_path / "round"
        d.mkdir()
        transcript.append_entries(d, [{"a": 1}])
        transcript.append_entries(d, [{"a": 1}, {"b": 2}])
        entries = transcript.read_jsonl(d / "transcript.jsonl")
        assert len(entries) == 2

    def test_rewrite_refused(self, tmp_path):
        d = tmp_path / "round"
        d.mkdir()
        transcript.append_entries(d, [{"a": 1}, {"b": 2}])
        with pytest.raises(transcript.AppendOnlyError):
            transcript.append_entries(d, [{"a": 1}])

    def test_rendered_markdown_created(self, tmp_path):
        d = tmp_path / "round"
        d.mkdir()
        transcript.append_entries(d, [{"seat": "echo", "type": "turn", "status": "completed"}])
        md = (d / "transcript.md").read_text()
        assert "# Round Table Transcript" in md


class TestBuslink:
    def test_token_from_env_only(self, monkeypatch):
        monkeypatch.setenv("BUS_TOKEN", "secret-token")
        link = buslink.BusLink(offline=True)
        assert link.token == "secret-token"

    def test_offline_queues_outbox(self, tmp_path):
        link = buslink.BusLink(offline=True, state_dir=tmp_path)
        link.publish_sync("committee.message.created", {"round_id": "r", "text": "hi"})
        outbox = list(link._read_outbox())
        assert len(outbox) == 1
        assert outbox[0].topic == "committee.message.created"
