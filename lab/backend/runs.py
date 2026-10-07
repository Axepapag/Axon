"""Real E0 run adapter; execution remains subject to trusted acceptance gates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import threading
import uuid

from core.manifests import E0_REFERENCE_GRAPH
from curriculum.schema import CURRICULUM_VERSION, validate_episode
from curriculum.splits import check_manifest, curriculum_sha256
from .store import canonical


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


class RunError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def supported_graph(graph):
    try:
        reference = {node['component_type']: node for node in E0_REFERENCE_GRAPH['nodes']}
        found = graph.get('nodes', [])
        if len(found) != len(reference) or len({n['component_type'] for n in found}) != len(reference): return False
        for node in found:
            expected = reference.get(node['component_type'])
            if expected is None or node['component_version'] != expected['component_version']: return False
            if {**expected['config'], **node.get('config', {})} != expected['config']: return False
        def edges(value):
            types = {n['node_id']: n['component_type'] for n in value['nodes']}
            return sorted((types[e['source']['node_id']], e['source']['port_id'],
                           types[e['destination']['node_id']], e['destination']['port_id']) for e in value['edges'])
        return edges(graph) == edges(E0_REFERENCE_GRAPH)
    except (KeyError, TypeError):
        return False


def artifact(root):
    base = root / 'State/curriculum_v1/e0-first'
    if not (base/'manifest.json').exists():
        raise RunError(503, 'curriculum_unavailable', 'The frozen E0 curriculum is unavailable.')
    try:
        manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
        episodes = [json.loads(line) for line in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        errors = check_manifest(manifest, episodes)
        errors += [error for episode in episodes for error in validate_episode(episode)]
        if errors: raise ValueError('; '.join(errors[:5]))
        digest = curriculum_sha256(episodes)
        dataset_id, curriculum_id = 'ds-e0-first-' + digest[:16], 'e0-first-' + digest[:16]
        splits = {bucket['split']: bucket['episode_ids'] for bucket in manifest['buckets']}
        dataset = {'dataset_id': dataset_id, 'name': 'E0 starter exercises', 'version': CURRICULUM_VERSION,
                   'dataset_hash': digest, 'native95_status': 'validated', 'conversion_status': 'not_converted',
                   'splits': {key: {'count': len(ids)} for key, ids in splits.items()},
                   'heldout_policy': manifest['rule'], 'manifest_hash': manifest['manifest_sha256']}
        curriculum = {'curriculum_id': curriculum_id, 'name': 'E0 first', 'version': CURRICULUM_VERSION,
                      'curriculum_hash': digest, 'dataset_id': dataset_id, 'stages': ['char_copy', 'delayed_recall', 'control']}
        return dataset, curriculum, episodes, splits
    except RunError: raise
    except Exception as exc:
        raise RunError(503, 'curriculum_invalid', 'Frozen curriculum verification failed.', {'reason': str(exc)}) from exc


class RunService:
    def __init__(self, registry, root, gate=None):
        self.registry, self.root = registry, Path(root)
        # Server-owned injection, never an HTTP/client bypass. Production default
        # stays closed until acceptance and independent restore evidence land.
        self.gate = gate or (lambda: {'authorized': False, 'reasons': [
            'E0 learning/response acceptance, complete recovery, independent backup and operator acceptance remain pending.']})
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='axon-run')
        self.active = None
        self.closing = False
        for run in registry.items('run'):
            if run['lifecycle_state'] in ('starting', 'running', 'pausing', 'resuming', 'stopping'):
                run.update(lifecycle_state='failed', failure_reason='Service restarted during execution; no automatic resume is claimed.')
                registry.put('run', run['run_id'], run, replace=True)
                self.event(run['run_id'], 'lifecycle', {'state': 'failed', 'reason': run['failure_reason']})

    def event(self, identity, kind, payload):
        return self.registry.append_run_event(identity, {'timestamp': now(), 'type': kind, 'payload': payload})

    def manifests(self):
        try:
            dataset, curriculum, _, _ = artifact(self.root)
            return [dataset], [curriculum], None
        except RunError as exc:
            return [], [], exc.message

    def get(self, identity):
        run = self.registry.get('run', identity)
        if not run: raise RunError(404, 'run_not_found', 'No such run.')
        run['readiness'] = self.gate.for_run(run) if hasattr(self.gate, 'for_run') else self.gate()
        actions = {'created': ['stop'], 'running': ['pause', 'stop', 'checkpoint'],
                   'paused': ['stop'], 'failed': ['stop']}.get(run['lifecycle_state'], [])
        if run['readiness']['authorized'] and run['lifecycle_state'] in ('created', 'paused'):
            actions.append('start' if run['lifecycle_state'] == 'created' else 'resume')
        run['allowed_actions'] = actions
        return run

    def prior(self, command_id, spec):
        try: return self.registry.prior_command(command_id, spec)
        except ValueError as exc: raise RunError(409, 'command_conflict', str(exc)) from exc

    def identity(self, spec):
        value = spec.get('command_id')
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise RunError(422, 'invalid_command', 'Supply a unique command_id.')
        return value

    def create(self, spec):
        command_id = self.identity(spec)
        settings = spec.get('training_settings') or {}
        if not isinstance(settings, dict) or set(settings) - {'epochs', 'learning_rate'}:
            raise RunError(422, 'invalid_settings', 'Supported settings: epochs and learning_rate.')
        epochs, rate, seed = settings.get('epochs', 1), settings.get('learning_rate', .001), spec.get('seed', 1)
        if type(epochs) is not int or not 1 <= epochs <= 100 or type(seed) is not int or not 0 <= seed < 2**32:
            raise RunError(422, 'invalid_settings', 'Epochs must be 1..100 and seed an unsigned 32-bit integer.')
        if type(rate) not in (int, float) or not 0 < rate <= .1:
            raise RunError(422, 'invalid_settings', 'Learning rate must be above zero and at most 0.1.')
        normalized = {**{key: value for key, value in spec.items() if key != 'command_id'}, 'kind': 'create_run'}
        with self.lock:
            prior = self.prior(command_id, normalized)
            if prior: return prior
            architecture = self.registry.get('architecture', spec.get('architecture_id', ''))
            if not architecture or not supported_graph(architecture):
                raise RunError(422, 'unsupported_architecture', 'Register the current exact E0 D512 reference graph first.')
            if spec.get('architecture_version') != architecture['version'] or spec.get('architecture_hash') != architecture['architecture_hash']:
                raise RunError(409, 'architecture_mismatch', 'Architecture version/hash differs from its registered identity.')
            dataset, curriculum, _, _ = artifact(self.root)
            if spec.get('dataset_id') != dataset['dataset_id'] or spec.get('curriculum_id') != curriculum['curriculum_id']:
                raise RunError(409, 'curriculum_mismatch', 'Select the current verified E0 curriculum and matching dataset.')
            if spec.get('provider') != 'local' or spec.get('device') not in ('cpu', 'cuda:0') or spec.get('split', 'train') != 'train':
                raise RunError(422, 'unsupported_execution', 'This adapter supports local CPU/CUDA0 training split only.')
            identity = str(uuid.uuid4())
            run = {'schema_version': 'axon-lab-api-v1', 'run_id': identity, 'lifecycle_state': 'created',
                   'architecture_id': architecture['architecture_id'], 'architecture_version': architecture['version'],
                   'architecture_hash': architecture['architecture_hash'], 'dataset_id': dataset['dataset_id'],
                   'dataset_hash': dataset['dataset_hash'], 'dataset_version': dataset['version'],
                   'curriculum_id': curriculum['curriculum_id'], 'curriculum_hash': curriculum['curriculum_hash'],
                   'curriculum_version': curriculum['version'], 'device': spec['device'], 'provider': 'local',
                   'seed': seed, 'training_settings': {'epochs': epochs, 'learning_rate': rate}, 'split': 'train',
                   'step': 0, 'epoch': 0, 'next_episode': 0, 'created_at': now(), 'failure_reason': None,
                   'pause_boundary': 'after_current_episode', 'latest_checkpoint_id': None, 'snapshot_id': None}
            operation = {'schema_version': 'axon-lab-api-v1', 'operation_id': str(uuid.uuid4()), 'kind': 'create_run',
                         'status': 'completed', 'run_id': identity, 'finished_at': now()}
            self.registry.run_command(command_id, normalized, operation, run)
            self.event(identity, 'lifecycle', {'state': 'created', 'execution_authorized': self.gate()['authorized']})
            return operation

    def command(self, identity, spec):
        command_id = self.identity(spec)
        action = spec.get('command')
        if action not in ('start', 'pause', 'resume', 'stop', 'checkpoint'):
            raise RunError(422, 'invalid_command', 'Unknown run command.')
        normalized = {'kind': 'run_command', 'run_id': identity, 'command': action}
        with self.lock:
            prior = self.prior(command_id, normalized)
            if prior: return prior
            run = self.get(identity)
            if self.closing and action in ('start', 'resume'):
                raise RunError(503, 'service_stopping', 'The service is stopping.')
            if run.get('pending_operation'):
                raise RunError(409, 'command_pending', 'Wait for the current command to reach its episode boundary.')
            if action in ('start', 'resume') and not run['readiness']['authorized']:
                raise RunError(409, 'execution_blocked', 'Execution acceptance and backup gates have not cleared.', run['readiness'])
            if action not in run['allowed_actions']:
                raise RunError(409, 'invalid_transition', 'This command is unavailable in the current lifecycle state.')
            if action in ('start', 'resume'):
                if self.active is not None: raise RunError(409, 'run_busy', 'Another run owns the execution worker.')
                import torch
                if run['device'] == 'cuda:0' and not torch.cuda.is_available():
                    raise RunError(409, 'device_unavailable', 'Selected CUDA device is unavailable; no fallback is allowed.')
                run['lifecycle_state'] = 'starting' if action == 'start' else 'resuming'
            else:
                run['lifecycle_state'] = {'pause': 'pausing', 'stop': 'stopping'}.get(action, run['lifecycle_state'])
            operation = {'schema_version': 'axon-lab-api-v1', 'operation_id': str(uuid.uuid4()), 'kind': 'run_command',
                         'status': 'queued', 'command': action, 'run_id': identity, 'created_at': now()}
            run['pending_operation'] = operation['operation_id']
            self.registry.run_command(command_id, normalized, operation, run)
            self.event(identity, 'lifecycle', {'state': run['lifecycle_state'], 'command': action, 'boundary': 'episode'})
            if action in ('start', 'resume'):
                self.active = identity
                self.pool.submit(self.work, identity, action == 'resume')
            elif self.active != identity:
                run['lifecycle_state'] = 'stopped'
                self.finish(run, operation)
            return operation

    def finish(self, run, operation):
        operation.update(status='completed', finished_at=now())
        run.pop('pending_operation', None)
        self.registry.complete_run_operation(run, operation, {'timestamp': now(), 'type': 'lifecycle',
                                             'payload': {'state': run['lifecycle_state']}})
        if hasattr(self.gate,'preserve_registry'): self.gate.preserve_registry(self.registry.path)

    def work(self, identity, resume):
        import random
        import numpy as np
        import torch
        from core.e0_two_state import E0TwoStateCore
        from runtime.heart.host import HeartHost
        from runtime.heart.host_e0 import E0Loop
        host = None
        try:
            run = self.get(identity)
            dataset, _, episodes, splits = artifact(self.root)
            if dataset['dataset_hash'] != run['dataset_hash']: raise ValueError('Dataset changed since run creation.')
            selected = [ep for ep in episodes if ep['episode_id'] in set(splits['train'])]
            root = self.registry.path.parent/'runs'/identity
            torch.manual_seed(run['seed']); random.seed(run['seed']); np.random.seed(run['seed'])
            host = HeartHost(root, consolidator_ids=('consolidator',))
            host.start()
            core = E0TwoStateCore()
            loop = E0Loop(root, host, core, selected, device=run['device'])
            optimizer = torch.optim.Adam(core.parameters(), lr=run['training_settings']['learning_rate'])
            loop.register_optimizer(optimizer)
            loop.bind_dataset('e0-first', curriculum_version=run['curriculum_version'],
                              curriculum_sha256=dataset['dataset_hash'], split_manifest_sha256=dataset['manifest_hash'])
            # Count actual successful optimizer calls without inventing loss.
            counter = {'steps': run['step']}
            optimizer.register_step_post_hook(lambda *_: counter.update(steps=counter['steps'] + 1))
            if resume:
                # Fresh-start lease events precede restoring the old host epoch;
                # do not mix them into that epoch's continued observations.
                host.drain_events()
                self.restore(run, loop, optimizer)
            with self.lock:
                current = self.get(identity)
                if current['lifecycle_state'] in ('starting', 'resuming'):
                    current['lifecycle_state'] = 'running'
                    self.finish(current, self.registry.get('operation', current['pending_operation']))
            total = len(selected) * run['training_settings']['epochs']
            source_seq = host.snapshot()['event_seq'] if resume else 0
            source_epoch = host.snapshot()['epoch']
            for position in range(run['next_episode'], total):
                with self.lock:
                    current = self.get(identity)
                    if self.closing:
                        current['latest_checkpoint_id'] = self.save(current, loop, optimizer)
                        current['lifecycle_state'] = 'paused'
                        if current.get('pending_operation'):
                            self.finish(current, self.registry.get('operation', current['pending_operation']))
                        else:
                            self.registry.put('run', identity, current, replace=True)
                            self.event(identity, 'lifecycle', {'state': 'paused', 'reason': 'service_shutdown'})
                        break
                    if current['lifecycle_state'] in ('pausing', 'stopping'):
                        self.boundary(current, loop, optimizer, counter['steps'], position)
                        break
                row = loop.run_episode(selected[position % len(selected)], train=True, optimizer=optimizer, split='train')
                row = {**row, 'measurement_scope': 'teacher_forced_training'}
                events = host.drain_events()
                for event in events:
                    if event['epoch'] != source_epoch:
                        self.event(identity, 'source_gap', {'reason': 'source_epoch_changed', 'previous_epoch': source_epoch, 'epoch': event['epoch']})
                        source_epoch, source_seq = event['epoch'], 0
                    if event['seq'] <= source_seq:
                        self.event(identity, 'source_gap', {'reason': 'repeated_source_ordinal', 'epoch': source_epoch, 'ordinal': event['seq']})
                        continue
                    if event['seq'] > source_seq + 1:
                        self.event(identity, 'source_gap', {'after_ordinal': source_seq, 'next_ordinal': event['seq'], 'epoch': event['epoch']})
                    source_seq = event['seq']
                    self.event(identity, 'heart.' + event['type'], {'source_ordinal': event['seq'], 'source_epoch': event['epoch'], **event['payload']})
                with self.lock:
                    current = self.get(identity)
                    current.update(step=counter['steps'], next_episode=position+1, epoch=(position+1)//len(selected), latest_result=row)
                    telemetry = row.get('training', {})
                    current['metrics'] = {'loss': telemetry.get('objective_loss', telemetry.get('final_loss')), 'loss_samples': telemetry.get('loss_samples', [])[-256:],
                                          'measurement_scope': 'teacher_forced_training'}
                    self.observe(current, loop, position+1, len(selected))
                    self.registry.put('run', identity, current, replace=True)
                    self.event(identity, 'episode_result', row)
                    if current.get('pending_operation'):
                        self.boundary(current, loop, optimizer, counter['steps'], position+1)
                        if current['lifecycle_state'] in ('paused', 'stopped'): break
            else:
                with self.lock:
                    current = self.get(identity)
                    current['latest_checkpoint_id'] = self.save(current, loop, optimizer)
                    current['lifecycle_state'] = 'completed'
                    if current.get('pending_operation'): self.finish(current, self.registry.get('operation', current['pending_operation']))
                    else:
                        self.registry.put('run', identity, current, replace=True)
                        self.event(identity, 'lifecycle', {'state': 'completed'})
                        if hasattr(self.gate,'preserve_registry'): self.gate.preserve_registry(self.registry.path)
        except Exception as exc:
            with self.lock:
                current = self.get(identity)
                current.update(lifecycle_state='failed', failure_reason=str(exc))
                if current.get('pending_operation'):
                    op = self.registry.get('operation', current.pop('pending_operation'))
                    op.update(status='failed', error={'code': 'run_failed', 'message': str(exc)}, finished_at=now())
                    self.registry.put('operation', op['operation_id'], op, replace=True)
                self.registry.put('run', identity, current, replace=True)
                self.event(identity, 'lifecycle', {'state': 'failed', 'reason': str(exc)})
        finally:
            try:
                if host is not None: host.stop()
            finally:
                with self.lock: self.active = None

    def boundary(self, run, loop, optimizer, steps, position):
        run.update(step=steps, next_episode=position)
        run['latest_checkpoint_id'] = self.save(run, loop, optimizer)
        if run['lifecycle_state'] in ('pausing', 'stopping'):
            run['lifecycle_state'] = 'paused' if run['lifecycle_state'] == 'pausing' else 'stopped'
        self.finish(run, self.registry.get('operation', run['pending_operation']))

    def observe(self, run, loop, next_episode, count):
        import torch
        snapshot_id = str(uuid.uuid4())
        tensors = {}
        for name, state in (('reasoning', loop.reasoning_state), ('response', loop.response_state)):
            values = state.detach().cpu().flatten()
            tensors[name] = {'tensor_id': name, 'name': name, 'semantic_role': name + '_state',
                'dtype': 'float32', 'shape': list(state.shape), 'device': str(state.device),
                'snapshot_id': snapshot_id, 'step': run['step'], 'values': values.tolist(),
                'statistics': {'min': float(values.min()), 'max': float(values.max()), 'mean': float(values.mean()),
                               'std': float(values.std(unbiased=False)), 'norm': float(torch.linalg.vector_norm(values))}}
        position = {'curriculum_id': run['curriculum_id'], 'curriculum_version': run['curriculum_version'],
                    'curriculum_hash': run['curriculum_hash'], 'dataset_id': run['dataset_id'],
                    'dataset_version': run['dataset_version'], 'dataset_hash': run['dataset_hash'], 'split': 'train',
                    'epoch': next_episode//count, 'episode_id': None, 'episode_index': next_episode % count,
                    'step_index': None, 'character_offset': None, 'emission_offset': None,
                    'phase': 'episode_boundary', 'optimizer_step': run['step']}
        snapshot = {'snapshot_id': snapshot_id, 'tensors': tensors, 'draft': loop.host.private_draft(),
                    'committed_response': loop.host.committed_text('response_draft'), 'heart': loop.host.inspect()}
        event = self.event(run['run_id'], 'snapshot', {'snapshot_id': snapshot_id, 'boundary': 'episode'})
        snapshot['last_observed_sequence'] = event['sequence']
        self.registry.put('run_snapshot', run['run_id'], snapshot, replace=True)
        run.update(snapshot_id=snapshot_id, last_observed_sequence=event['sequence'], execution_cursor={'schema_version': 'axon-lab-execution-cursor-v2',
                   'status': 'captured', 'snapshot_id': snapshot_id, 'position': position,
                   'rng': {'status': 'unavailable', 'state_ref': None, 'sha256': None}})

    def save(self, run, loop, optimizer):
        import torch
        identity = 'cp-' + str(uuid.uuid4())
        directory = loop.save(identity)
        torch.save({'optimizer': optimizer.state_dict(), 'cuda_rng': torch.cuda.get_rng_state_all() if run['device'] == 'cuda:0' else []}, directory/'lab_optimizer.pt')
        metadata = {'schema': 'axon-lab-episode-checkpoint-v1', 'run': run, 'field_id': loop.host.head_field_id,
                    'generation': loop.host.generation, 'files': {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in directory.rglob('*') if p.is_file()}}
        (directory/'lab_manifest.json').write_text(canonical(metadata), encoding='utf-8')
        preserved = self.gate.preserve_checkpoint(directory,run['run_id']) if hasattr(self.gate,'preserve_checkpoint') else None
        saved = {'checkpoint_id': identity, 'run_id': run['run_id'], 'parent_id': run.get('latest_checkpoint_id'),
                 'step': run['step'], 'architecture_hash': run['architecture_hash'], 'data_hash': run['dataset_hash'],
                 'completeness': True, 'boundary': 'episode', 'mid_episode_resume': False,
                 'directory': str(directory), 'manifest_sha256': hashlib.sha256((directory/'lab_manifest.json').read_bytes()).hexdigest(),
                 'backup_state': 'local_copy_verified_cloud_pending' if preserved else 'not_verified',
                 'backup_copy': preserved, 'restore_verification': None, 'created_at': now()}
        self.registry.put('checkpoint', identity, saved)
        self.event(run['run_id'], 'checkpoint', {key: value for key, value in saved.items() if key != 'directory'})
        return identity

    def restore(self, run, loop, optimizer):
        import torch
        saved = self.registry.get('checkpoint', run['latest_checkpoint_id'])
        directory = Path(saved['directory'])
        manifest_bytes = (directory/'lab_manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != saved['manifest_sha256']: raise ValueError('Checkpoint manifest checksum mismatch.')
        metadata = json.loads(manifest_bytes)
        for key in ('architecture_hash', 'dataset_hash', 'curriculum_hash', 'device', 'next_episode', 'step'):
            if metadata['run'][key] != run[key]: raise ValueError('Checkpoint run identity/cursor mismatch: ' + key)
        for name, digest in metadata['files'].items():
            path = (directory/name).resolve()
            path.relative_to(directory.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Checkpoint checksum mismatch: ' + name)
        if loop.host.head_field_id != metadata['field_id'] or loop.host.generation != metadata['generation']:
            raise ValueError('Heart advanced past checkpoint; exact resume refused.')
        if 'lab_optimizer.pt' not in metadata['files']:
            raise ValueError('Legacy checkpoint requires explicit migration before resume.')
        extra = torch.load(directory/'lab_optimizer.pt', map_location='cpu', weights_only=True)
        report = loop.load(saved['checkpoint_id'])
        optimizer.load_state_dict(extra['optimizer'])
        if run['device'] == 'cuda:0': torch.cuda.set_rng_state_all(extra['cuda_rng'])
        saved['restore_verification'] = {'status': 'restored', 'boundary': 'episode', 'host_report': report['host_restore'], 'at': now()}
        self.registry.put('checkpoint', saved['checkpoint_id'], saved, replace=True)

    def close(self):
        with self.lock:
            self.closing = True
        self.pool.shutdown(wait=True)
