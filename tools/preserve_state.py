"""Consistent private Axon runtime snapshot and hash-checked clean restore."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sqlite3, zipfile
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repository',type=Path,required=True); parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    staging=args.output/'snapshot'; staging.mkdir(exist_ok=True)
    with sqlite3.connect(args.registry) as source, sqlite3.connect(staging/'registry.sqlite3') as target:
        source.backup(target)
        assert target.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    shutil.copytree(args.repository/'State/curriculum_v1/e0-first',staging/'State/curriculum_v1/e0-first',dirs_exist_ok=True)
    shutil.copytree(args.trial/'organism/e0/memory-mechanism-trial',staging/'checkpoint',dirs_exist_ok=True)
    for name in ('plan.json','report.json'): shutil.copy2(args.trial/name,staging/name)
    files={p.relative_to(staging).as_posix():digest(p) for p in staging.rglob('*') if p.is_file()}
    (staging/'backup_manifest.json').write_text(json.dumps({'schema':'axon-private-backup-v1','files':files},indent=2),encoding='utf-8')
    archive=args.output/'axon-runtime-20261007.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as handle:
        for path in staging.rglob('*'):
            if path.is_file(): handle.write(path,path.relative_to(staging).as_posix())
    restored=args.output/'restored'; restored.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        for info in handle.infolist():
            target=(restored/info.filename).resolve()
            if not target.is_relative_to(restored.resolve()): raise ValueError('Unsafe archive path')
        handle.extractall(restored)
    assert all(digest(restored/name)==value for name,value in files.items())
    with sqlite3.connect(restored/'registry.sqlite3') as db: assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    from lab.backend.runs import artifact
    artifact(restored)
    from core.e0_two_state import load_checkpoint
    load_checkpoint(restored/'checkpoint/core.pt',map_location='cpu')
    result={'archive':str(archive),'sha256':digest(archive),'md5':hashlib.md5(archive.read_bytes()).hexdigest(),
            'bytes':archive.stat().st_size,'restore':{'path':str(restored),'file_count':len(files),'hashes_match':True,'sqlite_integrity':'ok','core_load':True,'curriculum_verified':True}}
    (args.output/'local_restore.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
