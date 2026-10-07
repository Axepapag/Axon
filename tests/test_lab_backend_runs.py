from copy import deepcopy
from contextlib import contextmanager
import json
import threading
import time

from fastapi.testclient import TestClient
import pytest
import torch

from core.manifests import E0_REFERENCE_GRAPH
from curriculum.presets import materialize
from lab.backend.app import create_app
from lab.backend.store import Registry
from runtime.heart.host_e0 import E0Loop


def prepared(client):
    graph = {**deepcopy(E0_REFERENCE_GRAPH), 'command_id': 'architecture'}
    # Canvas node IDs are opaque; topology must not depend on reference names.
    replacements = {n['node_id']: 'canvas-' + str(i) for i, n in enumerate(graph['nodes'])}
    for node in graph['nodes']: node['node_id'] = replacements[node['node_id']]
    for edge in graph['edges']:
        for point in ('source', 'destination'): edge[point]['node_id'] = replacements[edge[point]['node_id']]
    architecture = client.post('/api/v1/architectures', json=graph).json()
    dataset = client.get('/api/v1/datasets').json()['items'][0]
    curriculum = client.get('/api/v1/curricula').json()['items'][0]
    spec = {'command_id': 'create', 'architecture_id': architecture['architecture_id'],
            'architecture_version': architecture['version'], 'architecture_hash': architecture['architecture_hash'],
            'dataset_id': dataset['dataset_id'], 'curriculum_id': curriculum['curriculum_id'],
            'device': 'cpu', 'provider': 'local', 'seed': 10}
    response = client.post('/api/v1/runs', json=spec)
    assert response.status_code == 202, response.text
    return response.json()['run_id'], spec


def wait(client, identity, state):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        run = client.get('/api/v1/runs/' + identity).json()
        if run['lifecycle_state'] == state: return run
        if run['lifecycle_state'] == 'failed': pytest.fail(str(run))
        time.sleep(.02)
    pytest.fail('Boundary not reached: ' + str(run))


def test_prepare_and_gated_commands_are_durable_idempotent_and_bound(tmp_path):
    root = tmp_path/'project'
    materialize('e0-first', root/'State/curriculum_v1/e0-first')
    db = tmp_path/'db'
    with TestClient(create_app(root=root, state_path=db, core_probe=lambda: [])) as client:
        identity, spec = prepared(client)
        assert client.post('/api/v1/runs', json=spec).json()['run_id'] == identity
        changed = {**spec, 'seed': 20}
        assert client.post('/api/v1/runs', json=changed).status_code == 409
        blocked = client.post(f'/api/v1/runs/{identity}/commands', json={'command': 'start', 'command_id': 'start'})
        assert blocked.status_code == 409 and blocked.json()['code'] == 'execution_blocked'
        assert client.get('/api/v1/runs/' + identity).json()['allowed_actions'] == ['stop']
        wrong = {**spec, 'command_id': 'wrong', 'architecture_hash': '0'*64}
        assert client.post('/api/v1/runs', json=wrong).status_code == 409
        stop = {'command': 'stop', 'command_id': 'stop'}
        first = client.post(f'/api/v1/runs/{identity}/commands', json=stop).json()
        assert first['status'] == 'completed'
        assert client.post(f'/api/v1/runs/{identity}/commands', json=stop).json() == first
        with client.stream('GET', f'/api/v1/runs/{identity}/events', headers={'Accept': 'text/event-stream'}) as stream:
            assert 'data:' in stream.read().decode()
    with TestClient(create_app(root=root, state_path=db, core_probe=lambda: [])) as client:
        assert client.get('/api/v1/runs/' + identity).json()['lifecycle_state'] == 'stopped'
        events = client.get(f'/api/v1/runs/{identity}/events').json()['items']
        assert [e['sequence'] for e in events] == list(range(1, len(events)+1))
        tail = client.get(f'/api/v1/runs/{identity}/events?after_sequence=1').json()['items']
        assert tail == events[1:]


def test_real_loop_pause_restart_resume_stop_and_checked_checkpoint(tmp_path, monkeypatch):
    # A server-owned acceptance fixture unlocks only this isolated test; HTTP
    # callers have no gate override and no valuable artifacts are trained.
    torch.set_num_threads(1)
    root = tmp_path/'project'
    materialize('e0-first', root/'State/curriculum_v1/e0-first')
    db = tmp_path/'db'
    entered, release = threading.Event(), threading.Event()
    real = E0Loop.run_episode
    def bounded(self, *args, **kwargs):
        entered.set()
        assert release.wait(20)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(E0Loop, 'run_episode', bounded)
    gate = lambda: {'authorized': True, 'reasons': ['isolated test fixture']}
    app = create_app(root=root, state_path=db, core_probe=lambda: [], execution_gate=gate)
    with TestClient(app) as client:
        identity, _ = prepared(client)
        assert client.post(f'/api/v1/runs/{identity}/commands', json={'command': 'start', 'command_id': 'start'}).status_code == 202
        assert entered.wait(20)
        pause = {'command': 'pause', 'command_id': 'pause'}
        op = client.post(f'/api/v1/runs/{identity}/commands', json=pause)
        assert op.status_code == 202
        assert client.post(f'/api/v1/runs/{identity}/commands', json={'command': 'checkpoint', 'command_id': 'overlap'}).status_code == 409
        release.set()
        paused = wait(client, identity, 'paused')
        assert paused['next_episode'] == 1 and paused['step'] > 0
        operation = client.get('/api/v1/operations/' + op.json()['operation_id']).json()
        assert operation['status'] == 'completed'
        cp = client.get('/api/v1/checkpoints/' + paused['latest_checkpoint_id']).json()
        assert cp['completeness'] is True and cp['mid_episode_resume'] is False
        from pathlib import Path
        optimizer_path = Path(cp['directory'])/'lab_optimizer.pt'
        original = optimizer_path.read_bytes()
        optimizer_path.write_bytes(b'tampered')
        with pytest.raises(ValueError, match='checksum mismatch'):
            app.state.run_service.restore(paused, None, None)
        optimizer_path.write_bytes(original)
        bundle = json.loads((Path(cp['directory'])/'lab_manifest.json').read_text())
        assert 'optimizer.pt' in bundle['files'] and 'lab_optimizer.pt' in bundle['files']
        assert 'rng/torch_cpu' in bundle['files'] and 'rng/numpy' in bundle['files']
        torch.load(Path(cp['directory'])/'optimizer.pt', weights_only=True)['param_groups']
        assert paused['metrics']['loss'] is not None and len(paused['metrics']['loss_samples']) > 0
        assert paused['metrics']['measurement_scope'] == 'teacher_forced_training'
        assert paused['execution_cursor']['schema_version'] == 'axon-lab-execution-cursor-v2'
        from jsonschema import Draft202012Validator
        schema_path = Path(__file__).resolve().parents[1]/'lab/contracts/execution-cursor-v2.schema.json'
        Draft202012Validator(json.loads(schema_path.read_text())).validate(paused['execution_cursor'])
        snapshot = client.get(f'/api/v1/runs/{identity}/tensors').json()['items']
        assert len(snapshot) == 2
        coherent = client.get(f'/api/v1/runs/{identity}/snapshot').json()
        assert coherent['snapshot_id'] == paused['snapshot_id']
        assert 'values' not in coherent['tensors']['reasoning']
        assert coherent['last_observed_sequence'] == paused['last_observed_sequence']
        part = client.get(f'/api/v1/runs/{identity}/tensors/reasoning?count=16').json()
        assert len(part['values']) == 16 and part['total_count'] == 512
        assert client.get(f'/api/v1/runs/{identity}/tensors/reasoning?count=257').status_code == 422
        assert client.get(f'/api/v1/runs/{identity}/tensors/reasoning?snapshot_id=old').status_code == 409
    # Fresh backend + restored optimizer/loop, same episode-boundary cursor.
    entered.clear(); release.clear()
    with TestClient(create_app(root=root, state_path=db, core_probe=lambda: [], execution_gate=gate)) as client:
        response = client.post(f'/api/v1/runs/{identity}/commands', json={'command': 'resume', 'command_id': 'resume'})
        assert response.status_code == 202
        assert entered.wait(20)
        assert client.post(f'/api/v1/runs/{identity}/commands', json={'command': 'stop', 'command_id': 'stop'}).status_code == 202
        release.set()
        stopped = wait(client, identity, 'stopped')
        assert stopped['next_episode'] == 2 and stopped['step'] > paused['step']
        assert client.get('/api/v1/checkpoints/' + cp['checkpoint_id']).json()['restore_verification']['status'] == 'restored'
        events = client.get(f'/api/v1/runs/{identity}/events').json()['items']
        assert [e['sequence'] for e in events] == list(range(1, len(events)+1))
        assert any(e['type'].startswith('heart.') for e in events)


def test_changed_materialized_artifact_refuses_listing(tmp_path):
    root = tmp_path/'project'
    materialize('e0-first', root/'State/curriculum_v1/e0-first')
    path = root/'State/curriculum_v1/e0-first/episodes.jsonl'
    episodes = path.read_text().splitlines()
    item = json.loads(episodes[0]); item['expected']['text'] = 'altered'
    episodes[0] = json.dumps(item)
    path.write_text('\n'.join(episodes)+'\n')
    with TestClient(create_app(root=root, state_path=tmp_path/'db', core_probe=lambda: [])) as client:
        assert client.get('/api/v1/datasets').json()['items'] == []
        assert client.get('/api/v1/curricula').json()['unavailable_reason']


def test_pause_state_and_command_completion_publish_atomically(tmp_path, monkeypatch):
    registry = Registry(tmp_path/'atomic-db')
    registry.put('run', 'run', {'run_id': 'run', 'lifecycle_state': 'running'})
    registry.put('operation', 'operation', {'operation_id': 'operation', 'status': 'queued'})
    ready, release = threading.Event(), threading.Event()
    original = registry.connect
    @contextmanager
    def held_before_commit():
        with original() as db:
            yield db
            if threading.current_thread().name == 'completion-writer':
                ready.set()
                assert release.wait(10)
    monkeypatch.setattr(registry, 'connect', held_before_commit)
    thread = threading.Thread(name='completion-writer', target=lambda: registry.complete_run_operation(
        {'run_id': 'run', 'lifecycle_state': 'paused'},
        {'operation_id': 'operation', 'status': 'completed'},
        {'type': 'lifecycle', 'payload': {'state': 'paused'}}))
    thread.start()
    try:
        assert ready.wait(10)
        # Reads during the held commit see BOTH old values; never paused/queued.
        assert registry.get('run', 'run')['lifecycle_state'] == 'running'
        assert registry.get('operation', 'operation')['status'] == 'queued'
        assert registry.run_events('run', 0) == []
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert registry.get('run', 'run')['lifecycle_state'] == 'paused'
    assert registry.get('operation', 'operation')['status'] == 'completed'
    assert registry.run_events('run', 0)[0]['payload']['state'] == 'paused'
