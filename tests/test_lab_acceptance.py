import hashlib, json
from pathlib import Path
from fastapi.testclient import TestClient
from curriculum.presets import materialize
from lab.backend.acceptance import Acceptance, source_hash
from lab.backend.app import create_app
from lab.backend.runs import artifact


def fixture(root, state, phase='validation'):
    materialize('e0-first',root/'State/curriculum_v1/e0-first')
    (root/'core').mkdir(); (root/'core/example.py').write_text('initial\n')
    state.mkdir()
    proofs={}
    for name in ('learning_cpu','learning_cuda','recovery','backup','operator'):
        path=state/(name+'.json'); path.write_text(json.dumps({'passed':True,'destination':'test-only','artifact':{},'restore':{}}))
        proofs[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if phase=='validation': proofs.pop('operator')
    value={'schema':'axon-lab-acceptance-v1','source_hash':source_hash(root),'dataset_hash':artifact(root)[0]['dataset_hash'],
           'proofs':proofs,'phase':phase,'devices':['cuda:0'],'scope':'isolated gate fixture','validation_run_ids':['named'], 'evidence_id':'test'}
    (state/'acceptance.json').write_text(json.dumps(value))
    return Acceptance(root,state),value


def test_missing_or_tampered_proof_stays_closed(tmp_path):
    gate,_=fixture(tmp_path/'root',tmp_path/'state')
    assert gate()['authorized']
    (tmp_path/'state/recovery.json').write_text('{"passed":false}')
    assert not gate()['authorized']
    assert 'checksum' in gate()['reasons'][0]


def test_source_change_invalidates_acceptance(tmp_path):
    gate,_=fixture(tmp_path/'root',tmp_path/'state','accepted')
    assert gate()['authorized']
    (tmp_path/'root/core/example.py').write_text('changed\n')
    assert not gate()['authorized']


def test_validation_is_named_one_epoch_selected_device_only(tmp_path):
    gate,_=fixture(tmp_path/'root',tmp_path/'state')
    run={'run_id':'named','device':'cuda:0','training_settings':{'epochs':1}}
    assert gate.for_run(run)['authorized']
    assert not gate.for_run({**run,'run_id':'other'})['authorized']
    assert not gate.for_run({**run,'device':'cpu'})['authorized']
    assert not gate.for_run({**run,'training_settings':{'epochs':2}})['authorized']
    assert next(c for c in gate.checks() if c['id']=='operator_controls')['status']=='in_progress'


def test_http_cannot_authorize_execution(tmp_path):
    with TestClient(create_app(root=tmp_path/'project',state_path=tmp_path/'state/db',core_probe=lambda:[])) as client:
        assert client.post('/api/v1/acceptance',json={'passed':True,'authorized':True}).status_code==503
        assert client.get('/api/v1/readiness').json()['training_authorized'] is False
        assert client.get('/api/v1/capabilities').json()['feature_flags']['training'] is False


def test_preserved_checkpoint_restores_canonical_heart_in_new_root(tmp_path):
    from tools.restore_checkpoint import restore_complete_checkpoint
    from runtime.heart.host import HeartHost
    from runtime.heart.valve import ValveEnvelope
    gate,value=fixture(tmp_path/'root',tmp_path/'state','accepted')
    value['checkpoint_backup_root']=str(tmp_path/'private-backup')
    gate.path.write_text(json.dumps(value))
    organism=tmp_path/'original'
    host=HeartHost(organism,consolidator_ids=('consolidator',));host.start()
    try:
        host.submit_ingress(ValveEnvelope(valve_id='user_ingress',source_id='external_user',payload='A',provenance='backup-test',envelope_type='text/plain'))
        host.beat()
        assert host.committed_text('user_input')=='A'
        directory=organism/'e0/cp-test';directory.mkdir(parents=True)
        host.save_checkpoint(directory/'host_checkpoint.json')
        copy=Path(gate.preserve_checkpoint(directory,'run-test'))
        manifest=json.loads((copy/'restore_manifest.json').read_text())
        assert all(hashlib.sha256((copy/name).read_bytes()).hexdigest()==digest for name,digest in manifest['files'].items())
        restored=tmp_path/'independent-restore'
        restore=restore_complete_checkpoint(copy,restored)
    finally:host.stop()
    recovered=HeartHost(restored,consolidator_ids=('consolidator',));recovered.start()
    try:
        report=recovered.load_checkpoint(Path(restore['checkpoint'])/'host_checkpoint.json')
        assert report['behind_head'] is False
        assert recovered.committed_text('user_input')=='A'
    finally:recovered.stop()
    (copy/'host_checkpoint.json').write_text('{"tampered":true}')
    import pytest
    with pytest.raises(ValueError,match='checksum'):
        restore_complete_checkpoint(copy,tmp_path/'rejected-restore')
    assert not (tmp_path/'rejected-restore').exists()
