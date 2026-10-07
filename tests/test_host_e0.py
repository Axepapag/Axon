"""Tests for the E0 two-state loop over the HeartHost coordinator.

Everything runs on CPU with small cores and small generated curriculum
episodes.  Deterministic control paths use the loop's own ``control_override``
test seam (or a wrapped ``core.step`` when a test has to die mid-tick); the
character emission always stays the core's own argmax.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from core.e0_two_state import (
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_WAIT,
    E0TwoStateCore,
)
from curriculum.generators import generate_episode
from curriculum.metrics import RESULTS_FIELDS
from curriculum.schema import CURRICULUM_VERSION, validate_episode
from curriculum.splits import curriculum_sha256
from runtime.heart.errors import HostStateError
from runtime.heart.host import HeartHost
from runtime.heart.host_e0 import (
    CORE_CHECKPOINT_FILENAME,
    CURSOR_FILENAME,
    CURSOR_SCHEMA,
    HOST_CHECKPOINT_FILENAME,
    LOSS_SAMPLE_CAP,
    OPTIMIZER_CHECKPOINT_FILENAME,
    PHASE_OBSERVE,
    PHASE_RESPOND,
    RNG_DIRNAME,
    E0Loop,
)
from substrate.native import ALPHABET, encode_ids

CORE_KWARGS = {"width": 256, "hidden_size": 16, "max_chunk": 8}

ROW_EXTRAS = ("phases", "generation", "response_publication")


def _episodes() -> list[dict]:
    return [
        generate_episode("char_copy", 3, {"length": 1}),
        generate_episode("char_copy", 4, {"length": 2}),
        generate_episode("control", 7, {"scenario": "commit", "length": 2}),
        generate_episode("control", 8, {"scenario": "wait"}),
        generate_episode("control", 9, {"scenario": "end"}),
    ]


def _long_episode(length: int) -> dict:
    """A synthetic, schema-valid char_copy episode with a long expected text."""

    text = ("ab" * length)[:length]
    episode = {
        "episode_id": f"long-char-copy-{length}",
        "family": "char_copy",
        "seed": 0,
        "steps": [{"kind": "input", "text": text[:4], "control": None}],
        "query": "copy the text",
        "expected": {"text": text, "control": CONTROL_COMMIT},
        "facts": [text],
        "delay_ticks": 0,
        "distractor_count": None,
        "difficulty": {"length": 4},
        "split_key": "char_copy:copy:long",
    }
    assert validate_episode(episode) == []
    return episode


def _session(root: Path, episodes: list[dict], *, seed: int = 1234):
    torch.manual_seed(seed)
    host = HeartHost(root, consolidator_ids=("consolidator",))
    host.start()
    core = E0TwoStateCore(**CORE_KWARGS)
    loop = E0Loop(root, host, core, episodes, response_tick_budget=8)
    return host, core, loop


def _force_control(core: E0TwoStateCore, decision: int, *, counter: dict | None = None):
    """Wrap ``core.step`` so the control head always decides ``decision``."""

    original = core.step

    def step(char_ids, reasoning_state, response_state):
        if counter is not None:
            counter["ticks"] = counter.get("ticks", 0) + 1
        new_reasoning, new_response, char_logits, control_logits = original(
            char_ids, reasoning_state, response_state
        )
        forced = torch.full_like(control_logits, -1e9)
        forced[..., decision] = 0.0
        return new_reasoning, new_response, char_logits, forced

    return step


def _counting_step(core: E0TwoStateCore, counter: dict):
    original = core.step

    def step(char_ids, reasoning_state, response_state):
        counter["ticks"] = counter.get("ticks", 0) + 1
        return original(char_ids, reasoning_state, response_state)

    return step


def _journal_commits(host: HeartHost) -> list[dict]:
    events = [
        json.loads(line)
        for line in host.branch.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    return [event for event in events if event["event"] == "commit"]


# ------------------------------------------------------------------ end to end


def test_episodes_run_end_to_end_and_produce_results_rows(tmp_path: Path) -> None:
    episodes = _episodes()
    host, core, loop = _session(tmp_path, episodes)
    rows = []
    total_commits = 0
    for episode in episodes:
        rows.append(loop.run_episode(episode))
        total_commits += loop.commits  # per-episode counter, accumulated here

    assert len(rows) == len(episodes)
    for row, episode in zip(rows, episodes):
        assert list(row) == list(RESULTS_FIELDS) + list(ROW_EXTRAS)
        assert row["episode_id"] == episode["episode_id"]
        assert row["family"] == episode["family"]
        assert row["split"] == "train"
        assert row["curriculum_version"] == CURRICULUM_VERSION
        assert row["curriculum_sha256"] == curriculum_sha256(episodes)
        assert row["difficulty"] == episode["difficulty"]
        assert row["seed"] == episode["seed"]
        assert row["architecture_id"] == "axon.core_reasoning_gru"
        assert row["checkpoint_id"] is None
        assert row["prediction_control"] in (
            CONTROL_WAIT,
            CONTROL_COMMIT,
            CONTROL_END,
        )
        assert 1 <= row["phases"]["respond"] <= 8
        assert row["generation"]["mode"] == "independent_generation"
        assert row["phases"]["observe"] > 0
        assert set(row["metrics"]) == {
            "exact_match",
            "per_char_accuracy",
            "binding_error",
            "obsolete_error",
            "control_correct",
            "invalid_content",
            "wait_confusion",
        }
        assert row["metrics"]["invalid_content"] is False
        assert all(character in ALPHABET for character in row["prediction_text"])

    # Every canonical commit is the loop's own; an untrained core may emit none.
    assert host.generation == total_commits
    assert len(_journal_commits(host)) == total_commits
    host.stop()


def test_training_inputs_are_shifted_and_end_has_its_own_tick(tmp_path: Path) -> None:
    episode = dict(_episodes()[1], expected={"text": "ab", "control": CONTROL_COMMIT})
    host, core, loop = _session(tmp_path, [episode])
    observed = []
    original = core.step
    def recording_step(ids, reasoning, response):
        observed.append(int(ids.item()))
        return original(ids, reasoning, response)
    core.step = recording_step
    row = loop.run_episode(episode, train=True, optimizer=torch.optim.SGD(core.parameters(), lr=0.01))
    # First prediction gets no answer character; final tick predicts only END.
    assert observed[-3:] == [95, encode_ids("a")[0], encode_ids("b")[0]]
    assert row["prediction_text"] == "ab"
    assert row["generation"]["mode"] == "teacher_forced_training"
    assert row["generation"]["termination"] == "end"
    assert row["phases"]["respond"] == 3
    assert row["training"]["optimizer_steps"] == 1
    host.stop()


def test_response_loss_reaches_observed_memory(tmp_path: Path) -> None:
    episode = _episodes()[1]
    host, core, loop = _session(tmp_path, [episode])
    captured = []
    original = core.step
    def capture(ids, reasoning, response):
        result = original(ids, reasoning, response)
        if not captured:
            result[0].retain_grad()
            result[1].retain_grad()
            captured.extend(result[:2])
        return result
    core.step = capture
    loop.run_episode(episode, train=True, optimizer=torch.optim.SGD(core.parameters(), lr=0.01))
    assert all(tensor.grad is not None and torch.count_nonzero(tensor.grad) > 0 for tensor in captured)
    assert not loop.reasoning_state.requires_grad and not loop.response_state.requires_grad
    host.stop()


def test_generation_inputs_and_length_do_not_depend_on_expected_answer(tmp_path: Path) -> None:
    episode = _episodes()[1]
    host, core, loop = _session(tmp_path, [episode])
    calls = []
    original = core.step
    def step(ids, reasoning, response):
        calls.append(int(ids.item()))
        r, s, chars, controls = original(ids, reasoning, response)
        chars = torch.full_like(chars, -1e9)
        chars[:, encode_ids("x")[0]] = 0
        return r, s, chars, controls
    core.step = step
    loop.control_override = lambda phase, pos, logits: CONTROL_END if phase == PHASE_RESPOND and pos == 2 else CONTROL_COMMIT
    first = loop.run_episode(episode)
    first_calls = calls[:]
    calls.clear()
    changed = dict(episode, expected={"text": "some entirely different longer answer", "control": CONTROL_WAIT})
    second = loop.run_episode(changed)
    assert first["prediction_text"] == second["prediction_text"] == "xx"
    assert first["generation"] == second["generation"]
    assert calls == first_calls
    assert calls[-3:] == [95, encode_ids("x")[0], encode_ids("x")[0]]
    assert first["generation"]["termination"] == "end"
    assert first["prediction_control"] == CONTROL_COMMIT
    host.stop()


def test_generation_budget_exhaustion_is_explicit(tmp_path: Path) -> None:
    episode = _episodes()[0]
    host, core, loop = _session(tmp_path, [episode])
    loop.control_override = lambda phase, pos, logits: CONTROL_WAIT
    row = loop.run_episode(episode)
    assert row["generation"]["termination"] == "budget_exhausted"
    assert row["generation"]["response_ticks"] == 8
    assert not host.snapshot()["episode_ended"]
    assert host.private_draft() is None and loop.commits == 0
    host.stop()


def test_training_refuses_held_out_split_before_changes(tmp_path: Path) -> None:
    episode = _episodes()[0]
    host, core, loop = _session(tmp_path, [episode])
    before = {key: value.clone() for key, value in core.state_dict().items()}
    with pytest.raises(ValueError, match="only consume the train split"):
        loop.run_episode(episode, train=True, optimizer=torch.optim.SGD(core.parameters(), lr=0.01), split="test")
    assert all(torch.equal(value, core.state_dict()[key]) for key, value in before.items())
    assert host.generation == 0
    host.stop()


def test_incompatible_active_checkpoint_walk_fails_closed(tmp_path: Path) -> None:
    episode = _episodes()[0]
    host, core, loop = _session(tmp_path, [episode])
    observe = len(episode["steps"][0]["text"]) + len(episode["query"])
    original = core.step
    ticks = 0
    def interrupted(ids, reasoning, response):
        nonlocal ticks
        ticks += 1
        if ticks == observe + 2:
            loop.save("active")
            raise RuntimeError("interrupted")
        return original(ids, reasoning, response)
    loop.control_override = lambda phase, pos, logits: CONTROL_WAIT
    core.step = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        loop.run_episode(episode)
    host.stop()
    # An older active cursor does not name the new generation protocol.
    path = tmp_path / "e0/active" / CURSOR_FILENAME
    cursor = json.loads(path.read_text())
    cursor.pop("execution")
    path.write_text(json.dumps(cursor))
    restored_host, restored_core, restored = _session(tmp_path, [episode])
    restored.load("active")
    with pytest.raises(ValueError, match="execution protocol"):
        restored.run_episode(episode)
    assert restored_host.generation == 0
    restored_host.stop()


def test_split_argument_overrides_and_validates(tmp_path: Path) -> None:
    episodes = [_episodes()[0]]
    host, core, loop = _session(tmp_path, episodes)
    row = loop.run_episode(episodes[0], split="validation")
    assert row["split"] == "validation"
    with pytest.raises(ValueError):
        loop.run_episode(episodes[0], split="holdout")
    host.stop()


def test_episode_failing_validation_is_rejected_before_any_core_step(
    tmp_path: Path,
) -> None:
    episodes = [_episodes()[0]]
    host, core, loop = _session(tmp_path, episodes)
    counter: dict = {}
    core.step = _counting_step(core, counter)

    broken = dict(episodes[0], query="bad \u2019 quote")
    assert validate_episode(broken)  # the curriculum validator rejects it
    with pytest.raises(ValueError, match="failed curriculum validation"):
        loop.run_episode(broken)

    assert counter.get("ticks", 0) == 0
    assert host.generation == 0
    assert _journal_commits(host) == []
    host.stop()


# --------------------------------------------------------------- control paths


def test_wait_produces_zero_operations_and_zero_commits(tmp_path: Path) -> None:
    episodes = [_episodes()[0], _episodes()[3]]
    host, core, loop = _session(tmp_path, episodes)
    counter: dict = {}
    original_step = core.step
    core.step = _force_control(core, CONTROL_WAIT, counter=counter)

    row = loop.run_episode(episodes[0])

    assert counter["ticks"] > 0  # the whole walk ran
    assert loop.last_control == CONTROL_WAIT
    assert row["prediction_control"] == CONTROL_WAIT
    assert row["prediction_text"] == ""
    assert loop.commits == 0
    assert host.generation == 0
    assert _journal_commits(host) == []
    assert host.private_draft() is None
    assert host.spool.pending_count == 0

    # a WAIT-target episode (no expected text) stays zero-op in training too
    core.step = original_step
    loop.control_override = lambda phase, position, logits: CONTROL_WAIT
    optimizer = torch.optim.SGD(core.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in core.parameters()]
    wait_row = loop.run_episode(episodes[1], train=True, optimizer=optimizer)
    training = wait_row["training"]
    assert training["optimizer_steps"] == 1
    assert len(training["loss_samples"]) == 1
    assert wait_row["phases"]["respond"] == 1
    assert wait_row["prediction_text"] == ""
    assert loop.commits == 0
    assert host.generation == 0
    assert any(not torch.equal(parameter, start) for parameter, start in zip(core.parameters(), before))
    host.stop()


def test_end_finalizes_through_the_host(tmp_path: Path) -> None:
    episodes = _episodes()
    host, core, loop = _session(tmp_path, episodes)
    counter: dict = {}
    core.step = _force_control(core, CONTROL_END, counter=counter)

    row = loop.run_episode(episodes[0])
    observe = len(episodes[0]["steps"][0]["text"]) + len(episodes[0]["query"])
    assert counter["ticks"] == observe + 1  # observe first, then learned END
    assert row["prediction_control"] == CONTROL_END
    assert row["prediction_text"] == ""
    assert loop.commits == 0
    # the end-of-walk beat consumed the END control: the episode is finalized
    assert host.snapshot()["episode_ended"] is True
    with pytest.raises(HostStateError):
        host.submit_control("e0", "wait")

    # the next episode resets the host episode and runs normally
    core.step = _force_control(core, CONTROL_WAIT)
    second = loop.run_episode(episodes[1])
    assert second["prediction_control"] == CONTROL_WAIT
    assert host.snapshot()["episode_ended"] is False
    host.stop()


def test_committed_text_flows_only_through_the_consolidator(tmp_path: Path) -> None:
    episodes = [_episodes()[1]]
    host, core, loop = _session(tmp_path, episodes)
    counter: dict = {}
    loop.control_override = lambda phase, position, logits: CONTROL_END if phase==PHASE_RESPOND and position==3 else CONTROL_COMMIT

    row = loop.run_episode(episodes[0])

    assert loop.last_control == CONTROL_END
    assert row["phases"]["respond"] == 4
    assert row["generation"]["termination"] == "end"
    assert loop.commits == 1
    assert len(row["prediction_text"]) >= 1
    assert row["metrics"]["invalid_content"] is False
    # the canonical text exists only because the consolidator committed it
    assert host.committed_text("response_draft") == row["prediction_text"]
    commits = _journal_commits(host)
    assert len(commits) == loop.commits
    for event in commits:
        assert event["metadata"]["committer_id"] == "consolidator"
        assert event["metadata"]["proposal_id"]
    assert loop.last_commit_id == commits[-1]["delta_id"]
    assert host.generation == loop.commits  # the loop is the only writer
    host.stop()


def test_stage_stays_private_and_end_publishes_entire_response(tmp_path):
    episode=_episodes()[1]
    host,core,loop=_session(tmp_path,[episode])
    with torch.no_grad():
        core.char_head.weight.zero_();core.char_head.bias.fill_(-100)
        core.char_head.bias[encode_ids('A')[0]]=100
    real_finish=host.finish_response
    def finish(core_id):
        assert host.private_draft()['text']=='AA'
        assert host.committed_text()==''
        assert host.generation==0
        return real_finish(core_id)
    host.finish_response=finish
    loop.control_override=lambda phase,pos,_: CONTROL_END if phase==PHASE_RESPOND and pos==2 else CONTROL_COMMIT
    try:
        row=loop.run_episode(episode)
        assert host.committed_text()=='AA'
        assert host.generation==loop.commits==1
        assert row['response_publication']['status']=='published'
        assert _journal_commits(host)[0]['metadata']['committer_id']=='consolidator'
        host.finish_response=real_finish
        again=host.finish_response('e0')
        assert not again['committed'] and host.generation==1
    finally:host.stop()


def test_unfinished_text_stays_private_when_generation_budget_expires(tmp_path):
    episode=_episodes()[1]
    host,core,loop=_session(tmp_path,[episode])
    with torch.no_grad():
        core.char_head.weight.zero_();core.char_head.bias.fill_(-100)
        core.char_head.bias[encode_ids('A')[0]]=100
    loop.control_override=lambda *_: CONTROL_COMMIT
    try:
        row=loop.run_episode(episode)
        assert row['prediction_text']=='A'*8
        assert row['generation']['termination']=='budget_exhausted'
        assert row['response_publication']['status']=='private'
        assert host.committed_text()=='' and host.generation==loop.commits==0
    finally:host.stop()


def test_publication_crash_resume_does_not_duplicate_write_or_training(tmp_path):
    episode=dict(_episodes()[1],expected={'text':'AB','control':CONTROL_COMMIT})
    def build(path):
        host,core,loop=_session(path,[episode])
        opt=torch.optim.Adam(core.parameters(),lr=.001);loop.register_optimizer(opt)
        return host,core,loop,opt
    host,core,loop,opt=build(tmp_path/'reference')
    try:
        expected=loop.run_episode(episode,train=True,optimizer=opt)
        weights={k:v.clone() for k,v in core.state_dict().items()}
    finally:host.stop()
    host,core,loop,opt=build(tmp_path/'interrupted')
    real_step=core.step
    def step(*args):
        if loop._generation.get('response_ticks')==2:loop.save('before-end')
        return real_step(*args)
    core.step=step
    real_commit=host.commit
    def interrupted_commit(*args):
        real_commit(*args)
        raise RuntimeError('interrupted after response publication')
    host.commit=interrupted_commit
    try:
        with pytest.raises(RuntimeError,match='after response publication'):
            loop.run_episode(episode,train=True,optimizer=opt)
        assert host.committed_text()=='AB' and host.generation==1
    finally:host.stop()
    host,core,loop,opt=build(tmp_path/'interrupted')
    try:
        loaded=loop.load('before-end')
        assert loaded['host_restore']['behind_head']
        actual=loop.run_episode(episode,train=True,optimizer=opt)
        assert actual['training']==expected['training']
        assert all(torch.equal(weights[k],v) for k,v in core.state_dict().items())
        assert host.committed_text()=='AB' and host.generation==1
        assert len(_journal_commits(host))==1
        assert actual['response_publication']['status']=='unchanged'
    finally:host.stop()


# ---------------------------------------------------------------- restart paths


def test_save_and_load_restore_core_host_cursor_and_rng(tmp_path: Path) -> None:
    episodes = _episodes()
    host, core, loop = _session(tmp_path, episodes)
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    expected_torch_rng = torch.get_rng_state().clone()
    expected_py_rng = random.getstate()
    expected_numpy_rng = np.random.get_state()
    directory = loop.save("tag-1")

    cursor = json.loads((directory / CURSOR_FILENAME).read_text(encoding="utf-8"))
    assert set(cursor) == {
        "episode_index",
        "step_index",
        "torch_rng_state",
        "numpy_rng_state",
        "py_rng_state",
        "tag",
        "cursor",  # additive structured axon-lab-execution-cursor-v1 block
        "execution",
        "response_publication",
    }
    assert (directory / CORE_CHECKPOINT_FILENAME).exists()
    assert (directory / HOST_CHECKPOINT_FILENAME).exists()
    # no optimizer was registered: no optimizer artifact is invented
    assert not (directory / OPTIMIZER_CHECKPOINT_FILENAME).exists()
    rng_files = sorted(path.name for path in (directory / RNG_DIRNAME).iterdir())
    assert {"numpy", "python", "torch_cpu"} <= set(rng_files)
    assert all(
        name == "torch_cpu" or name.startswith("torch_cuda_") or name in ("numpy", "python")
        for name in rng_files
    )
    assert cursor["tag"] == "tag-1"
    assert cursor["cursor"]["schema"] == CURSOR_SCHEMA
    expected_epoch = host.snapshot()["epoch"]
    host.stop()

    resumed_host, resumed_core, resumed = _session(tmp_path, episodes)  # seeds RNG anew
    report = resumed.load("tag-1")

    assert report["tag"] == "tag-1"
    assert report["host_restore"]["behind_head"] is False
    assert resumed.checkpoint_id == "tag-1"
    assert resumed_host.snapshot()["epoch"] == expected_epoch  # epoch continued
    # the durable cursor's RNG context is restored, not the session's own
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert random.getstate() == expected_py_rng
    assert np.array_equal(np.random.get_state()[1], expected_numpy_rng[1])
    assert report["step_index"] == cursor["step_index"]
    assert report["cursor"] == cursor["cursor"]
    assert report["optimizer_restored"] is False
    resumed_host.stop()


def test_mid_episode_kill_resumes_bit_identically(tmp_path: Path) -> None:
    episodes = [_episodes()[1]]

    def build(root: Path):
        return _session(root, episodes)

    # uninterrupted reference run, same tag so the rows match exactly
    host_plain, core_plain, loop_plain = build(tmp_path / "plain")
    loop_plain.save("run")
    core_plain.step = _force_control(core_plain, CONTROL_COMMIT)
    row_plain = loop_plain.run_episode(episodes[0])
    host_plain.stop()

    # interrupted run: save mid-episode and die inside a tick
    host_killed, core_killed, loop_killed = build(tmp_path / "killed")
    loop_killed.save("run")
    calls = {"ticks": 0}
    forced_step = _force_control(core_killed, CONTROL_COMMIT)

    observe_ticks = len(episodes[0]["steps"][0]["text"]) + len(episodes[0]["query"])
    def counting_dying_step(char_ids, reasoning_state, response_state):
        calls["ticks"] += 1
        if calls["ticks"] == observe_ticks + 3:
            loop_killed.save("run")  # the kill point, mid-episode
            raise RuntimeError("simulated kill mid-episode")
        return forced_step(char_ids, reasoning_state, response_state)

    core_killed.step = counting_dying_step
    with pytest.raises(RuntimeError, match="simulated kill"):
        loop_killed.run_episode(episodes[0])
    saved_cursor = json.loads(
        (tmp_path / "killed" / "e0" / "run" / CURSOR_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    mid_step = saved_cursor["step_index"]
    walk_length = observe_ticks + 8
    assert 0 < mid_step < walk_length
    assert loop_killed.commits == 0
    assert host_killed.private_draft()['text']
    assert host_killed.committed_text()==''  # the prefix is still private
    host_killed.stop()

    # a brand-new session loads the mid-episode checkpoint and continues with
    # the same forced control behaviour as the uninterrupted reference run
    host_resumed, core_resumed, loop_resumed = build(tmp_path / "killed")
    report = loop_resumed.load("run")
    assert report["step_index"] == mid_step
    core_resumed.step = _force_control(core_resumed, CONTROL_COMMIT)
    row_resumed = loop_resumed.run_episode(episodes[0])
    host_resumed.stop()

    assert row_resumed == row_plain
    # per-loop commit counters count one instance's own commits; the durable
    # draft text spans the killed and the resumed session
    assert row_resumed['prediction_text']
    assert loop_killed.commits + loop_resumed.commits == 0  # no END was chosen


# ---------------------------------------------------------------------- train


def test_train_changes_weights_and_infer_does_not(tmp_path: Path) -> None:
    episodes = [_episodes()[0]]
    host, core, loop = _session(tmp_path, episodes)
    loop.control_override = lambda phase, position, logits: CONTROL_COMMIT

    frozen_bank = core.native_cells.clone()
    before = [parameter.detach().clone() for parameter in core.parameters()]

    loop.run_episode(episodes[0], train=False)
    assert all(
        torch.equal(parameter, start)
        for parameter, start in zip(core.parameters(), before)
    )

    optimizer = torch.optim.SGD(core.parameters(), lr=0.1)
    row = loop.run_episode(episodes[0], train=True, optimizer=optimizer)
    assert any(
        not torch.equal(parameter, start)
        for parameter, start in zip(core.parameters(), before)
    )
    # the frozen substrate is never trained
    assert torch.equal(core.native_cells, frozen_bank)
    assert row["training"]["optimizer_steps"] == 1

    with pytest.raises(ValueError, match="requires the caller's optimizer"):
        loop.run_episode(episodes[0], train=True)
    host.stop()


def test_full_response_phase_supervises_every_expected_character(
    tmp_path: Path,
) -> None:
    episode = _episodes()[1]
    expected_text = episode["expected"]["text"]
    assert len(expected_text) == 2
    host, core, loop = _session(tmp_path, [episode])
    loop.control_override = lambda phase, position, logits: CONTROL_COMMIT
    optimizer = torch.optim.SGD(core.parameters(), lr=0.05)

    row = loop.run_episode(episode, train=True, optimizer=optimizer)

    training = row["training"]
    assert training["optimizer_steps"] == 1
    assert len(training["loss_samples"]) == len(expected_text) + 1
    assert training["final_loss"] == training["loss_samples"][-1]
    assert all(isinstance(value, float) for value in training["loss_samples"])
    assert all(value > 0.0 for value in training["loss_samples"])
    assert row["phases"] == {"observe": row["phases"]["observe"], "respond": 3}
    assert row["phases"]["observe"] > 0
    # teacher-forced emission: the draft (and the canonical field) hold exactly
    #the expected response, each character committed through the consolidator
    assert row["prediction_text"] == expected_text
    assert host.committed_text("response_draft") == expected_text
    assert loop.commits == 1
    assert host.generation == loop.commits
    # the last control the core predicted is the argmax it produced, not the target
    assert row["prediction_control"] == loop.last_control
    host.stop()


def test_premature_end_is_masked_in_the_respond_phase(tmp_path: Path) -> None:
    episode = _episodes()[1]
    expected_text = episode["expected"]["text"]
    host, core, loop = _session(tmp_path, [episode])
    seen: list[tuple[str, int, int]] = []

    def force_early_end(phase: str, position: int, logits):
        if phase != PHASE_RESPOND:
            seen.append((phase, position, None))
            return None
        decision = CONTROL_END if position < len(expected_text) - 1 else CONTROL_COMMIT
        seen.append((phase, position, decision))
        return decision

    loop.control_override = force_early_end
    optimizer = torch.optim.SGD(core.parameters(), lr=0.05)
    row = loop.run_episode(episode, train=True, optimizer=optimizer)

    # the seam overrode only the respond phase: END was predicted on every
    # non-final respond step and never executed, and the observation walk was
    # never overridden
    respond_calls = [item for item in seen if item[0] == PHASE_RESPOND]
    observe_calls = [item for item in seen if item[0] == PHASE_OBSERVE]
    assert respond_calls == [
        (PHASE_RESPOND, 0, CONTROL_END),
        (PHASE_RESPOND, 1, CONTROL_COMMIT),
        (PHASE_RESPOND, 2, CONTROL_COMMIT),
    ]
    assert observe_calls and all(decision is None for _p, _i, decision in observe_calls)
    assert len(row["training"]["loss_samples"]) == len(expected_text) + 1
    # the walk still consumed the whole target
    assert row["prediction_text"] == expected_text
    assert host.committed_text("response_draft") == expected_text
    assert loop.last_control == CONTROL_COMMIT
    # the final respond step executed END: the episode finalized through the host
    assert host.snapshot()["episode_ended"] is True
    with pytest.raises(HostStateError):
        host.submit_control("e0", "wait")
    host.stop()


def test_inference_runs_independent_response_phase(tmp_path: Path) -> None:
    episode = _episodes()[1]
    host, core, loop = _session(tmp_path, [episode])
    row = loop.run_episode(episode)

    assert "training" not in row
    assert row["phases"]["respond"] > 0
    assert row["generation"]["mode"] == "independent_generation"
    assert row["prediction_text"] != episode["expected"]["text"]  # no teacher forcing
    assert "training" not in row
    host.stop()


def test_loss_samples_are_bounded_to_the_last_256(tmp_path: Path) -> None:
    length = LOSS_SAMPLE_CAP + 4
    episode = _long_episode(length)
    host, core, loop = _session(tmp_path, [episode])
    loop.control_override = lambda phase, position, logits: CONTROL_COMMIT
    optimizer = torch.optim.SGD(core.parameters(), lr=0.02)

    row = loop.run_episode(episode, train=True, optimizer=optimizer)
    training = row["training"]

    assert training["optimizer_steps"] == 1
    assert len(training["loss_samples"]) == LOSS_SAMPLE_CAP
    assert training["final_loss"] == training["loss_samples"][-1]
    assert row["phases"]["respond"] == length + 1
    assert row["prediction_text"] == episode["expected"]["text"]
    assert host.committed_text("response_draft") == episode["expected"]["text"]
    host.stop()


def test_training_resume_restores_weights_optimizer_and_loss_tail(
    tmp_path: Path,
) -> None:
    episode = _episodes()[1]
    expected_length = len(episode["expected"]["text"])

    def build(root: Path):
        host, core, loop = _session(root, [episode])
        loop.control_override = lambda phase, position, logits: CONTROL_COMMIT
        optimizer = torch.optim.SGD(core.parameters(), lr=0.05, momentum=0.9)
        loop.register_optimizer(optimizer)
        return host, core, loop, optimizer

    # uninterrupted reference
    host_plain, core_plain, loop_plain, optimizer_plain = build(tmp_path / "plain")
    reference = loop_plain.run_episode(episode, train=True, optimizer=optimizer_plain)
    reference_losses = reference["training"]["loss_samples"]
    host_plain.stop()

    # interrupted: die inside the last respond step, saving mid-episode
    host_killed, core_killed, loop_killed, optimizer_killed = build(tmp_path / "killed")
    base_step = core_killed.step
    observe_ticks = len(episode["steps"][0]["text"]) + len(episode["query"])
    kill_at = observe_ticks + expected_length  # the final respond tick
    calls = {"ticks": 0}

    def dying_step(char_ids, reasoning_state, response_state):
        calls["ticks"] += 1
        if calls["ticks"] == kill_at:
            loop_killed.save("run")
            raise RuntimeError("simulated kill mid-training")
        return base_step(char_ids, reasoning_state, response_state)

    core_killed.step = dying_step
    with pytest.raises(RuntimeError, match="simulated kill"):
        loop_killed.run_episode(episode, train=True, optimizer=optimizer_killed)
    saved = json.loads(
        (tmp_path / "killed" / "e0" / "run" / CURSOR_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    done = saved["cursor"]["next"]["char_position"]  # respond chars already emitted
    assert 0 < done < expected_length
    assert (tmp_path / "killed" / "e0" / "run" / OPTIMIZER_CHECKPOINT_FILENAME).exists()
    host_killed.stop()

    # a fresh session resumes mid-episode with its own optimizer object
    host_resumed, core_resumed, loop_resumed, optimizer_resumed = build(
        tmp_path / "killed"
    )
    report = loop_resumed.load("run")
    assert report["optimizer_restored"] is True
    resumed = loop_resumed.run_episode(episode, train=True, optimizer=optimizer_resumed)
    host_resumed.stop()

    # identical subsequent loss samples: the resumed run replays the tail
    assert resumed["training"]["loss_samples"] == reference_losses
    assert resumed["training"]["optimizer_steps"] == 1
    # identical weights and identical optimizer state (momentum buffers included)
    for name, value in core_plain.state_dict().items():
        assert torch.equal(value, core_resumed.state_dict()[name]), name
    plain_state = optimizer_plain.state_dict()
    resumed_state = optimizer_resumed.state_dict()
    assert plain_state["param_groups"] == resumed_state["param_groups"]
    assert set(plain_state["state"]) == set(resumed_state["state"])
    for key, value in plain_state["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, resumed_state["state"][key][field]), field
    assert resumed["prediction_text"] == reference["prediction_text"]
    assert resumed["phases"] == reference["phases"]


def test_cursor_block_round_trips_and_carries_the_dataset_binding(
    tmp_path: Path,
) -> None:
    episodes = [_episodes()[0]]
    host, core, loop = _session(tmp_path, episodes)
    directory = loop.save("plain")
    plain = json.loads((directory / CURSOR_FILENAME).read_text(encoding="utf-8"))

    assert plain["cursor"] == {
        "schema": CURSOR_SCHEMA,
        "next": {
            "episode_index": None,
            "step_index": 0,
            "char_position": 0,
            "emission_count": 0,
        },
        "dataset": {
            "preset": None,
            "curriculum_version": None,
            "curriculum_sha256": None,
            "split_manifest_sha256": None,
        },
        "last_control": None,
        "episode_ended": False,
    }

    binding = loop.bind_dataset(
        "e0-first",
        curriculum_version=loop.curriculum_version,
        curriculum_sha256=loop.curriculum_sha256,
        split_manifest_sha256="a" * 64,
    )
    assert binding["preset"] == "e0-first"
    assert binding["curriculum_sha256"] == loop.curriculum_sha256
    with pytest.raises(ValueError):
        loop.bind_dataset("")
    directory = loop.save("bound")
    bound = json.loads((directory / CURSOR_FILENAME).read_text(encoding="utf-8"))
    assert bound["cursor"]["dataset"] == {
        "preset": "e0-first",
        "curriculum_version": CURRICULUM_VERSION,
        "curriculum_sha256": loop.curriculum_sha256,
        "split_manifest_sha256": "a" * 64,
    }
    host.stop()

    # loading restores the structured block, including the dataset binding
    resumed_host, resumed_core, resumed = _session(tmp_path, episodes)
    assert resumed.save("before-load")  # a loop with no binding writes nulls
    report = resumed.load("bound")
    assert report["cursor"] == bound["cursor"]
    assert resumed.save("after-load") and json.loads(
        (tmp_path / "e0" / "after-load" / CURSOR_FILENAME).read_text(encoding="utf-8")
    )["cursor"]["dataset"]["preset"] == "e0-first"
    resumed_host.stop()

