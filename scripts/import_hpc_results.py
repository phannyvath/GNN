"""Copy an extracted HPC study directory into the local dashboard archive.

No pickle/checkpoint is loaded. Inputs are copied, never executed. Existing runs
are not overwritten. run_status.json can describe queued/running/failed/interrupted
snapshots; absent status is reported as unknown, not inferred from process IDs.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from study_dashboard_data import enrich, load

ROOT=Path(__file__).resolve().parents[1]

def import_run(source, name, root=ROOT):
    if not name or Path(name).name!=name or name in ('.','..'):
        raise ValueError('Use a single directory name')
    source=Path(source).resolve()
    if not source.is_dir(): raise ValueError('Source must be an extracted study directory')
    destination=root/'results/imported'/name
    if destination.exists(): raise ValueError('Destination exists; use a new snapshot name')
    if source==destination or source in destination.parents: raise ValueError('Destination cannot be inside source')
    if any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Symlinks are not accepted; copy regular files')
    if not (source/'config.json').exists(): raise ValueError('Missing config.json')
    rows=load(source/'metrics.json',[])
    evidence=enrich(source,rows)
    if (source/'study_complete.json').exists() and not evidence['compatible']:
        raise ValueError('Completed study lacks consistent audit, fingerprints or model/seed coverage')
    checksums={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.log')}
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,destination)
    (destination/'import_record.json').write_text(json.dumps(dict(imported_utc=datetime.now(timezone.utc).isoformat(),source_directory=str(source),state=evidence['run_state'],file_sha256=checksums,verification='Structural consistency only; source audit claims retained. Import does not rerun prediction/checkpoint verification.'),indent=2))
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--name',required=True)
    a=p.parse_args(); print(import_run(a.source,a.name))
    print('Rebuild: .venv/bin/python scripts/build_study_dashboard.py')
