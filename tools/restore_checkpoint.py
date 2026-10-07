"""Hash-verified complete checkpoint restore to a new directory, never in place."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from tools.paths import filesystem_path, canonical_path


def restore_complete_checkpoint(backup: Path, destination: Path):
    backup=filesystem_path(backup).resolve(); destination=filesystem_path(destination).resolve()
    if destination.exists(): raise ValueError('Restore destination must be new; existing state is never overwritten.')
    manifest=json.loads((backup/'restore_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='axon-complete-checkpoint-backup-v1': raise ValueError('Unsupported backup schema.')
    actual={p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file() and p!=backup/'restore_manifest.json'}
    if actual!=set(manifest['files']): raise ValueError('Backup file inventory mismatch.')
    for name,digest in manifest['files'].items():
        path=(backup/name).resolve()
        if not path.is_relative_to(backup): raise ValueError('Unsafe backup path.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('Backup checksum mismatch: '+name)
    tag=manifest['checkpoint_tag']
    if Path(tag).name!=tag or tag in ('.','..'): raise ValueError('Unsafe checkpoint tag.')
    shutil.copytree(backup/'heart_state',destination)
    checkpoint=destination/'e0'/tag
    shutil.copytree(backup,checkpoint,ignore=lambda _,names:[n for n in names if n in ('heart_state','restore_manifest.json')])
    host_path=checkpoint/'host_checkpoint.json'
    original=host_path.read_bytes(); host=json.loads(original)
    # Explicit offline relocation: identity, generation, field IDs and memory
    # remain unchanged. A new checkpoint documents the new filesystem root.
    relative=Path(host['branch']['root']).relative_to(Path(manifest['original_run_root']))
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe branch relocation.')
    new_root=str(canonical_path(destination)/relative)
    old_root=host['branch']['root']; host['branch']['root']=new_root
    migrated=json.dumps(host,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n'
    (checkpoint/'host_checkpoint.source.json').write_bytes(original)
    host_path.write_bytes(migrated)
    host_path.with_name(host_path.name+'.sha256').write_text(hashlib.sha256(migrated).hexdigest()+'\n',encoding='ascii')
    lab_path=checkpoint/'lab_manifest.json'
    if lab_path.exists():
        lab=json.loads(lab_path.read_text(encoding='utf-8'))
        for name in ('host_checkpoint.json','host_checkpoint.json.sha256'):
            lab['files'][name]=hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
        lab_path.write_text(json.dumps(lab,sort_keys=True,separators=(',',':')),encoding='utf-8')
    report={'schema':'axon-checkpoint-relocation-v1','backup':str(backup),'destination':str(destination),'checkpoint':str(checkpoint),
        'files_verified':len(manifest['files']),'old_branch_root':old_root,'new_branch_root':new_root,
        'source_host_sha256':hashlib.sha256(original).hexdigest(),'relocated_host_sha256':hashlib.sha256(migrated).hexdigest(),
        'scope':'Complete Heart/E0 checkpoint restore; importing into the Lab run registry remains a separate integration.'}
    (destination/'restore_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backup',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(restore_complete_checkpoint(args.backup,args.destination)))


if __name__=='__main__':main()
