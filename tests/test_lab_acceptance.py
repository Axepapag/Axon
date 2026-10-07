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
