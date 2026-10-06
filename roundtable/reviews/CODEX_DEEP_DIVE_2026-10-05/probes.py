"""Isolated audit reproductions; never opens real memories or changes runtime State."""
from pathlib import Path
import importlib.util
import json
import sqlite3
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from substrate import substrate_1024 as substrate1024
from substrate.native import ALPHABET
from runtime.field import FieldSpan
from runtime.heart.valve import ValveEnvelope, primitive_valve_registry
from runtime.soul import SoulStore, SoulTransition, SoulLayer, SoulTemperature
import runtime.soul.store as soul_module


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_probes():
    results = {}
    vector = substrate1024.encode_text('A').astype(np.float64)
    before = vector.copy()
    vector[0, 0] = np.nextafter(vector[0, 0], np.inf)
    results['zero_tolerance_precision'] = {
        'input_changed': bool(np.any(vector != before)),
        'difference': float(np.max(np.abs(vector - before))),
        'decoded': substrate1024.decode_vectors(vector, tolerance=0.0),
    }
    cell = substrate1024.lane_bank()[0:1].astype(np.float64)
    cell[0, 0] = np.nextafter(cell[0, 0], np.inf)
    results['zero_tolerance_precision']['modified_cell_lift_accepted'] = bool(
        substrate1024.cells16_to_vectors(cell).shape == (1, 1024))

    payload = 'outside ' + chr(0x1F600)
    envelope = ValveEnvelope('user_ingress', 'external_user', payload,
                             'synthetic-audit', 'text/plain')
    decision = primitive_valve_registry().decide(envelope)
    results['ingress_character_guard'] = {'admitted': decision.admitted}
    try:
        FieldSpan(span_id='probe', text=payload, source='audit', provenance='synthetic')
        results['ingress_character_guard']['field_span'] = 'accepted'
    except Exception as exc:
        results['ingress_character_guard']['field_span'] = type(exc).__name__

    audit = load_file('isolated_character_audit', ROOT / 'tools/audit_characters.py')
    importer = load_file('isolated_memory_importer', ROOT / 'curator/import_d00_memories.py')
    with tempfile.TemporaryDirectory(prefix='axon-audit-only-') as temp:
        scratch = Path(temp)
        path = scratch / 'outside.txt'
        raw = b'alpha\r\nbeta\rGAMMA\vdelta\ffinish'
        path.write_bytes(raw)
        measured = audit.audit_file(path)
        results['character_audit'] = {
            'raw_outside_count': sum(chr(byte) not in ALPHABET for byte in raw),
            'reported_outside': measured['outside'],
            'records': measured['records'],
            'clean_records': measured['clean_records'],
        }
        path = scratch / 'malformed.jsonl'
        path.write_text('{"broken":"abc"\n', encoding='utf-8')
        measured = audit.audit_file(path)
        results['character_audit']['malformed_json_records'] = measured['records']
        results['character_audit']['malformed_json_clean'] = measured['clean_records']

        store = SoulStore.active(scratch / 'private-state')
        before = store.ensure_core(core_id='audit', architecture_id='audit-gru',
                                   parameter_generation='g0')
        branch = store.branch('audit')
        transition = SoulTransition(
            core_id=before.core_id, architecture_id=before.architecture_id,
            parameter_generation=before.parameter_generation,
            before_soul_id=before.soul_id, before_generation=before.generation,
            tick_uid='audit-tick', request_id='audit-request', phase='first',
            updates=(SoulLayer(SoulTemperature.HOT, b'synthetic-hot-state'),))
        original = soul_module._atomic_json
        def fail_before_head(path, value):
            if Path(path) == branch.head_path:
                raise OSError('synthetic failure after receipt before HEAD')
            return original(path, value)
        soul_module._atomic_json = fail_before_head
        try:
            try:
                branch.commit_transition(transition)
            except OSError:
                pass
        finally:
            soul_module._atomic_json = original
        result = {
            'receipt_exists': (branch.receipts_dir / (transition.transition_id + '.json')).exists(),
            'head_is_old_before_recovery': branch.load_head().soul_id == before.soul_id,
        }
        try:
            branch.recover()
            result['recovery'] = 'completed'
        except Exception as exc:
            result['recovery_error'] = type(exc).__name__
            result['recovery_message'] = str(exc)
        result['head_is_old_after_recovery'] = branch.load_head().soul_id == before.soul_id
        results['soul_receipt_before_head_crash'] = result

        db = scratch / 'synthetic.db'
        writer = sqlite3.connect(db)
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute('CREATE TABLE sample(value TEXT)')
        writer.commit()
        writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        writer.execute('INSERT INTO sample VALUES (?)', ('synthetic-new-record',))
        writer.commit()
        normal = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
        immutable = importer._open_readonly(db)
        try:
            results['wal_importer'] = {
                'normal_readonly_rows': normal.execute('SELECT COUNT(*) FROM sample').fetchone()[0],
                'importer_rows': immutable.execute('SELECT COUNT(*) FROM sample').fetchone()[0],
                'wal_exists': Path(str(db) + '-wal').exists(),
            }
        finally:
            normal.close()
            immutable.close()
            writer.close()
    return results


if __name__ == '__main__':
    results = run_probes()
    output = json.dumps(results, indent=2) + '\n'
    if len(sys.argv) > 1:
        with Path(sys.argv[1]).open('x', encoding='utf-8') as handle:
            handle.write(output)
    print(output)
