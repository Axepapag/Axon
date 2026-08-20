"""Stateful seat tests for the Round Table Orchestrator v1.1."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.table import cli, sessions, waker
from runtime.table.manifests import ManifestDir, load_manifest


FIXTURES = Path(__file__).parent / "fixtures" / "table"
BRIEF = FIXTURES / "test_brief.md"


def _write_wrapper(tmp_path: Path, name: str, code: str) -> Path:
    """Write a small Python wrapper script for a test seat."""
    wrapper = tmp_path / f"{name}.py"
    wrapper.write_text(code, encoding="utf-8")
    return wrapper


def _make_manifests_dir(tmp_path: Path, extra: dict[str, dict] | None = None) -> Path:
    """Copy the standard manifests and overlay any extra seat definitions."""
    d = tmp_path / "manifests"
    d.mkdir()
    for name in ("echo_a.json", "echo_b.json", "synth.json"):
        shutil.copy(ManifestDir / name, d / name)
    for name, data in (extra or {}).items():
        (d / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return d


class TestSessionStore:
    def test_round_trip(self, tmp_path):
        state = tmp_path / "State" / "table"
        sessions.set_session(state_root=state, seat_id="kimi", session_id="session_abc123")
        assert sessions.get_session(state_root=state, seat_id="kimi") == "session_abc123"

    def test_pinned_not_overwritten_by_capture(self, tmp_path):
        state = tmp_path / "State" / "table"
        sessions.set_session(state_root=state, seat_id="kimi", session_id="session_pinned", pinned=True)
        sessions.capture_session(state_root=state, seat_id="kimi", session_id="session_new")
        assert sessions.get_session(state_root=state, seat_id="kimi") == "session_pinned"
        assert sessions.is_pinned(state_root=state, seat_id="kimi") is True

    def test_unpinned_is_overwritten_by_capture(self, tmp_path):
        state = tmp_path / "State" / "table"
        sessions.set_session(state_root=state, seat_id="kimi", session_id="session_old", pinned=False)
        sessions.capture_session(state_root=state, seat_id="kimi", session_id="session_new")
        assert sessions.get_session(state_root=state, seat_id="kimi") == "session_new"

    def test_clear_unpinned(self, tmp_path):
        state = tmp_path / "State" / "table"
        sessions.set_session(state_root=state, seat_id="kimi", session_id="session_old", pinned=False)
        assert sessions.clear_session(state_root=state, seat_id="kimi", only_if_pinned=False) is True
        assert sessions.get_session(state_root=state, seat_id="kimi") is None

    def test_clear_pinned_respected(self, tmp_path):
        state = tmp_path / "State" / "table"
        sessions.set_session(state_root=state, seat_id="kimi", session_id="session_pinned", pinned=True)
        assert sessions.clear_session(state_root=state, seat_id="kimi", only_if_pinned=True) is False
        assert sessions.get_session(state_root=state, seat_id="kimi") == "session_pinned"


class TestManifestV2:
    def test_v1_manifest_still_loads(self):
        # echo.json has no v2 fields and must remain valid.
        manifest = load_manifest(ManifestDir / "echo.json")
        assert manifest.wake.resume_argv is None
        assert manifest.model is None
        assert manifest.parse_regex is None
        assert manifest.enabled is True

    def test_kimi_manifest_has_resume_and_regex(self):
        manifest = load_manifest(ManifestDir / "kimi.json")
        assert manifest.wake.resume_argv is not None
        assert "{session_id}" in " ".join(manifest.wake.resume_argv)
        assert manifest.parse_regex is not None
        assert "session_" in manifest.parse_regex

    def test_hermes_manifest_has_model_substitution(self):
        manifest = load_manifest(ManifestDir / "hermes.json")
        assert manifest.model == "glm-5.2:cloud"
        assert "{model}" in " ".join(manifest.wake.argv)

    def test_model_substitution_in_argv(self, tmp_path):
        state = tmp_path / "State" / "table"
        manifests_dir = _make_manifests_dir(tmp_path)

        wrapper_code = '''
import sys
idx = sys.argv.index("--model")
model = sys.argv[idx + 1]
print("```json")
print('{"type": "post", "text": "model=' + model + '", "stamp": "s"}')
print("```")
'''
        wrapper = _write_wrapper(tmp_path, "model_probe", wrapper_code)

        seat = {
            "client_id": "model_probe",
            "display_name": "Model Probe",
            "model": "glm-test-model",
            "wake": {
                "argv": ["python", str(wrapper), "--model", "{model}"],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "capabilities": ["test"],
            "identity_stamp": "M / m / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (manifests_dir / "model_probe.json").write_text(json.dumps(seat), encoding="utf-8")

        table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
        table.open_round(
            round_id="r1",
            brief_path=str(BRIEF),
            seats=["model_probe"],
            synthesizer="synth",
            max_cycles=1,
            per_turn_timeout_s=30,
            budgets={"max_invocations": 5, "board_view_events": 40, "board_view_chars": 24000},
            offline=True,
            officers=[],
        )
        result = table.step("r1")
        assert result["halted"] is False
        assert result["entry"]["actions"][0].get("text") == "model=glm-test-model"

    def test_model_placeholder_requires_model_value(self, tmp_path):
        d = tmp_path / "manifests"
        d.mkdir()
        bad = {
            "client_id": "bad",
            "display_name": "Bad",
            "wake": {
                "argv": ["python", "wrapper.py", "--model", "{model}", "{prompt}"],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "capabilities": [],
            "identity_stamp": "Bad / bad / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="model"):
            waker.RoundTable(manifests_dir=d)


class TestResumeArgvSelection:
    def test_uses_resume_argv_when_session_known(self, tmp_path):
        state = tmp_path / "State" / "table"
        manifests_dir = _make_manifests_dir(tmp_path)

        # Wrapper embeds argv[1] in the JSON text so we can tell which template was used.
        wrapper = _write_wrapper(
            tmp_path,
            "resume_probe",
            'import sys\nmode = sys.argv[1]\n'
            'print(\'```json\\n{"type": "post", "text": "mode=\' + mode + \'", "stamp": "p / m / d"}\\n```\')\n',
        )

        seat = {
            "client_id": "resume_probe",
            "display_name": "Resume Probe",
            "wake": {
                "argv": ["python", str(wrapper), "fresh"],
                "resume_argv": ["python", str(wrapper), "resume"],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "session": {"parse_regex": ""},
            "capabilities": ["test"],
            "identity_stamp": "Probe / probe / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (manifests_dir / "resume_probe.json").write_text(json.dumps(seat), encoding="utf-8")

        # Without a session id, fresh argv is used.
        table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
        table.open_round(
            round_id="r1",
            brief_path=str(BRIEF),
            seats=["resume_probe"],
            synthesizer="synth",
            max_cycles=2,
            per_turn_timeout_s=30,
            budgets={"max_invocations": 10, "board_view_events": 40, "board_view_chars": 24000},
            offline=True,
            officers=[],
        )
        result = table.step("r1")
        assert result["halted"] is False
        assert result["entry"]["status"] == "completed"
        assert result["entry"]["actions"][0].get("text") == "mode=fresh"

        # Pin a session id; next turn must use resume_argv.
        sessions.pin_session(state_root=state, seat_id="resume_probe", session_id="session_abc")
        result2 = table.step("r1")
        assert result2["entry"]["status"] == "completed"
        assert result2["entry"]["actions"][0].get("text") == "mode=resume"

    def test_session_capture_from_stdout(self, tmp_path):
        state = tmp_path / "State" / "table"
        manifests_dir = _make_manifests_dir(tmp_path)

        wrapper = _write_wrapper(
            tmp_path,
            "capture_probe",
            'import sys\nprint("To resume this session: probe -r session_captured_123")\n'
            'print(\'```json\\n{"type": "pass", "stamp": "p / m / d"}\\n```\')\n',
        )

        seat = {
            "client_id": "capture_probe",
            "display_name": "Capture Probe",
            "wake": {
                "argv": ["python", str(wrapper)],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "session": {"parse_regex": "To resume this session: probe -r (session_[a-z0-9_-]+)"},
            "capabilities": ["test"],
            "identity_stamp": "Probe / probe / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (manifests_dir / "capture_probe.json").write_text(json.dumps(seat), encoding="utf-8")

        table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
        table.open_round(
            round_id="r1",
            brief_path=str(BRIEF),
            seats=["capture_probe"],
            synthesizer="synth",
            max_cycles=1,
            per_turn_timeout_s=30,
            budgets={"max_invocations": 5, "board_view_events": 40, "board_view_chars": 24000},
            offline=True,
            officers=[],
        )
        assert sessions.get_session(state_root=state, seat_id="capture_probe") is None
        table.step("r1")
        assert sessions.get_session(state_root=state, seat_id="capture_probe") == "session_captured_123"
        assert sessions.is_pinned(state_root=state, seat_id="capture_probe") is False


class TestResumeFailureFallback:
    def test_broken_session_falls_back_to_fresh(self, tmp_path):
        state = tmp_path / "State" / "table"
        manifests_dir = _make_manifests_dir(tmp_path)

        # Resume mode exits nonzero; fresh mode succeeds.
        wrapper = _write_wrapper(
            tmp_path,
            "breakable",
            'import sys\n'
            'if sys.argv[1] == "resume":\n'
            '    print("broken session", file=sys.stderr)\n'
            '    sys.exit(1)\n'
            'print("fresh ok")\n'
            'print(\'```json\\n{"type": "pass", "stamp": "p / m / d"}\\n```\')\n',
        )

        seat = {
            "client_id": "breakable",
            "display_name": "Breakable",
            "wake": {
                "argv": ["python", str(wrapper), "fresh"],
                "resume_argv": ["python", str(wrapper), "resume"],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "session": {"parse_regex": ""},
            "capabilities": ["test"],
            "identity_stamp": "B / b / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (manifests_dir / "breakable.json").write_text(json.dumps(seat), encoding="utf-8")

        table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
        table.open_round(
            round_id="r1",
            brief_path=str(BRIEF),
            seats=["breakable"],
            synthesizer="synth",
            max_cycles=1,
            per_turn_timeout_s=30,
            budgets={"max_invocations": 5, "board_view_events": 40, "board_view_chars": 24000},
            offline=True,
            officers=[],
        )

        sessions.pin_session(state_root=state, seat_id="breakable", session_id="session_broken")
        result = table.step("r1")
        assert result["halted"] is False
        entry = result["entry"]
        assert entry["status"] == "completed"
        assert entry.get("session_resume_failed") is True
        # The broken unpinned session id is cleared after fallback.
        assert sessions.get_session(state_root=state, seat_id="breakable") is None

    def test_empty_stdout_resume_falls_back(self, tmp_path):
        state = tmp_path / "State" / "table"
        manifests_dir = _make_manifests_dir(tmp_path)

        wrapper = _write_wrapper(
            tmp_path,
            "empty_resume",
            'import sys\n'
            'if sys.argv[1] == "resume":\n'
            '    sys.exit(0)  # empty stdout counts as failed resume\n'
            'print("fresh ok")\n'
            'print(\'```json\\n{"type": "pass", "stamp": "p / m / d"}\\n```\')\n',
        )

        seat = {
            "client_id": "empty_resume",
            "display_name": "Empty Resume",
            "wake": {
                "argv": ["python", str(wrapper), "fresh"],
                "resume_argv": ["python", str(wrapper), "resume"],
                "prompt_via": "argv",
                "timeout_seconds": 30,
                "workdir": str(tmp_path),
            },
            "session": {"parse_regex": ""},
            "capabilities": ["test"],
            "identity_stamp": "E / e / 2026-07-03",
            "enabled": True,
            "notes": "",
        }
        (manifests_dir / "empty_resume.json").write_text(json.dumps(seat), encoding="utf-8")

        table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
        table.open_round(
            round_id="r1",
            brief_path=str(BRIEF),
            seats=["empty_resume"],
            synthesizer="synth",
            max_cycles=1,
            per_turn_timeout_s=30,
            budgets={"max_invocations": 5, "board_view_events": 40, "board_view_chars": 24000},
            offline=True,
            officers=[],
        )

        sessions.set_session(state_root=state, seat_id="empty_resume", session_id="session_empty")
        result = table.step("r1")
        assert result["entry"]["status"] == "completed"
        assert result["entry"].get("session_resume_failed") is True


class TestSessionCLI:
    def test_pin_unpin_seats(self, tmp_path, monkeypatch):
        state = tmp_path / "State" / "table"
        # Point the default state root to our temp path via monkeypatch on the module default.
        monkeypatch.setattr(sessions, "DefaultStateRoot", state)

        assert cli.main(["pin", "--seat", "kimi", "--session", "session_cli_123"]) == 0
        assert sessions.get_session(seat_id="kimi") == "session_cli_123"
        assert sessions.is_pinned(seat_id="kimi") is True

        # seats should list the pinned session.
        code = cli.main(["seats"])
        assert code == 0

        assert cli.main(["unpin", "--seat", "kimi"]) == 0
        assert sessions.is_pinned(seat_id="kimi") is False
        assert sessions.get_session(seat_id="kimi") == "session_cli_123"

    def test_pin_requires_session(self, tmp_path, monkeypatch):
        state = tmp_path / "State" / "table"
        monkeypatch.setattr(sessions, "DefaultStateRoot", state)
        with pytest.raises(SystemExit):
            cli.main(["pin", "--seat", "kimi"])
