"""Server-owned acceptance evidence; HTTP clients cannot publish or bypass it."""
from __future__ import annotations
import hashlib, json
from pathlib import Path


def source_hash(root: Path) -> str:
    entries = {}
    for folder in ('substrate', 'core', 'curriculum', 'runtime', 'lab/backend', 'lab/frontend'):
        for path in sorted((root/folder).rglob('*')):
            if path.suffix not in ('.py', '.js', '.css', '.html', '.npy') or '__pycache__' in path.parts or 'tests' in path.parts: continue
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    for name in ('tools/acceptance_memory.py', 'tools/acceptance_recovery.py', 'tools/preserve_state.py', 'tools/restore_checkpoint.py'):
        path = root/name
        if path.exists(): entries[name] = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


class Acceptance:
    def __init__(self, root: Path, state: Path):
        self.root, self.path = root, state/'acceptance.json'

    def evidence(self):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
            if value.get('schema') != 'axon-lab-acceptance-v1': raise ValueError('Unsupported acceptance schema.')
            if value.get('source_hash') != source_hash(self.root): raise ValueError('Source changed since acceptance; repeat affected checks.')
            from .runs import artifact
            if value.get('dataset_hash') != artifact(self.root)[0]['dataset_hash']: raise ValueError('Frozen exercises differ from the accepted version.')
            for item in value['proofs'].values():
                raw = Path(item['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != item['sha256']: raise ValueError('Acceptance proof checksum mismatch.')
                if json.loads(raw).get('passed') is not True: raise ValueError('An acceptance proof did not pass.')
            if not {'learning_cpu', 'learning_cuda', 'recovery', 'backup'} <= value['proofs'].keys(): raise ValueError('Required acceptance proofs are missing.')
            if value.get('phase') not in ('validation', 'accepted'): raise ValueError('Acceptance is incomplete.')
            if value['phase'] == 'accepted' and 'operator' not in value['proofs']: raise ValueError('Operator proof missing.')
            return value, None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, str(exc)

    def __call__(self):
        value, error = self.evidence()
        if not value: return {'authorized':False, 'mode':'blocked', 'reasons':[error or 'Acceptance evidence unavailable.']}
        validation = value['phase'] == 'validation'
        authorized = bool(value.get('validation_run_ids')) if validation else True
        return {'authorized':authorized, 'mode':'bounded_validation' if validation else 'accepted',
                'devices':value['devices'], 'scope':value['scope'],
                'reasons':['Only the named one-epoch operator-validation run can start.'] if validation else [],
                'evidence_id':value['evidence_id']}

    def for_run(self, run):
        result = self()
        if not result['authorized']: return result
        value, _ = self.evidence()
        if run['device'] not in value['devices']:
            return {**result,'authorized':False,'reasons':['Selected device has no accepted recovery/learning evidence.']}
        if value['phase'] == 'validation' and (run['run_id'] not in value.get('validation_run_ids',[]) or run['training_settings']['epochs'] != 1):
            return {**result,'authorized':False,'reasons':['This run is outside the bounded operator-validation scope.']}
        return result

    def checks(self):
        value, error = self.evidence()
        labels = {'core_runtime':'Measured CPU/GPU starter memory mechanism and exact Heart path.',
                  'dataset_split':'Frozen starter data and disjoint diagnostic train/validation/test contexts.',
                  'checkpoint_resume':'Bit-identical CPU/GPU interrupted-training recovery.',
                  'backup_restore':'Current source and private runtime artifact backup independently restored.',
                  'operator_controls':'Browser Start/Pause/Checkpoint/restart/Resume/Stop on CUDA.'}
        return [{'id':key, 'status':('in_progress' if key=='operator_controls' and value['phase']=='validation' else 'passed') if value else 'not_verified',
                 'reason':label if value else error, 'evidence':value['evidence_id'] if value else None} for key,label in labels.items()]

    def backup(self):
        value, error = self.evidence()
        if not value: return {'status':'not_verified','configured_destination':None,'verified_artifact':None,'restore_drill':None,'reason':error}
        report = json.loads(Path(value['proofs']['backup']['path']).read_text(encoding='utf-8'))
        return {'status':'verified','configured_destination':report['destination'],'verified_artifact':report['artifact'],
                'restore_drill':report['restore'],'reason':'Source clone and hash-verified private runtime restore passed; later checkpoints need their own preservation.'}

    def preserve_checkpoint(self, directory: Path, run_id: str):
        import shutil
        value, error = self.evidence()
        if not value: raise ValueError('Checkpoint preservation unavailable: '+str(error))
        target=Path(value['checkpoint_backup_root'])/run_id/directory.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(directory,target)
        for path in directory.rglob('*'):
            if path.is_file() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256((target/path.relative_to(directory)).read_bytes()).digest():
                raise ValueError('Checkpoint backup checksum mismatch.')
        # A host checkpoint references canonical branch objects. Preserve those
        # dependencies at this same episode boundary, before the next beat.
        run_root=directory.parent.parent
        context=target/'heart_state'
        def ignore(path,names):
            skipped={'heart.lock','lease.json'}
            if Path(path)==run_root: skipped.add('e0')
            return [name for name in names if name in skipped]
        shutil.copytree(run_root,context,ignore=ignore)
        for path in run_root.rglob('*'):
            relative=path.relative_to(run_root)
            if relative.parts[0]=='e0' or path.name in ('heart.lock','lease.json'): continue
            if path.is_file() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256((context/relative).read_bytes()).digest():
                raise ValueError('Canonical Heart backup checksum mismatch.')
        files={p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
        (target/'restore_manifest.json').write_text(json.dumps({'schema':'axon-complete-checkpoint-backup-v1','files':files,
            'original_run_root':str(run_root.resolve()),'checkpoint_tag':directory.name,
            'layout':'Restore heart_state as the run root and checkpoint artifacts under e0/'+directory.name+'. Recreate OS lease on start.'},indent=2),encoding='utf-8')
        return str(target)

    def preserve_registry(self, source: Path):
        import datetime as dt, sqlite3
        value,error=self.evidence()
        # Preparation and stopping an unstarted run remain available while the
        # execution gate is closed; no approved preservation target exists yet.
        if not value: return None
        target=Path(value['checkpoint_backup_root'])/'registry'/('snapshot-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.sqlite3')
        target.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(source) as src, sqlite3.connect(target) as dest:
            src.backup(dest)
            if dest.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('Registry backup integrity failure.')
        return str(target)
