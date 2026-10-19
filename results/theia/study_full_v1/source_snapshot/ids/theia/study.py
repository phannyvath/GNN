"""Reproducible study entry point: separate data, fresh workers, explicit resume."""
import argparse
import json
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import save_json, sha256
from .study_data import make_data, verify_data, validate_config

MODELS=('flash_node','magic_node','random_forest_node')
ROOT=Path(__file__).resolve().parents[2]


def code_hashes():
    paths=list((ROOT/'ids/theia').glob('*.py'))+[ROOT/'ids/models.py',ROOT/'ids/metrics.py',ROOT/'scripts/run_theia_study.sh',ROOT/'requirements.txt']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def verify_complete(path,fingerprint):
    complete=json.loads((path/'complete.json').read_text())
    if complete['fingerprint']!=fingerprint:raise ValueError('Completed worker data mismatch')
    for name,digest in complete['artifacts'].items():
        if sha256(path/name)!=digest:raise ValueError(f'Completed artifact modified: {path/name}')
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/theia_study_full.json');p.add_argument('--output');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',help='Explicitly resume this existing study, preserving config and data')
    a=p.parse_args();source=ROOT/'data/theia'
    if a.resume:
        if a.output or a.smoke:p.error('--resume cannot be combined with --output or --smoke')
        out=(ROOT/a.resume).resolve();c=json.loads((out/'config.json').read_text());validate_config(c)
        if json.loads((out/'code_fingerprints.json').read_text())!=code_hashes():raise ValueError('Source code changed since study creation; start a new run instead of silently mixing code')
        verify_data(out/'data',source,c)
        with (out/'resume_invocations.jsonl').open('a') as f:f.write(json.dumps({'argv':sys.argv,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
    else:
        c=json.loads((ROOT/a.config).read_text())
        if a.smoke:c.update(name='theia_study_smoke',role='real_data_smoke_NOT_FULL_STUDY',epochs=2,patience=2,rf_trees=10,smoke_nodes=4096)
        validate_config(c)
        out=(ROOT/a.output).resolve() if a.output else ROOT/'results/theia'/f"{c['name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        out.mkdir(parents=True,exist_ok=False)
        save_json(out/'config.json',c);save_json(out/'code_fingerprints.json',code_hashes())
        for relative in code_hashes():
            target=out/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        (out/'source_snapshot/configs').mkdir(exist_ok=True)
        save_json(out/'source_snapshot/configs/theia_study_full.json',c)
        save_json(out/'software_versions.json',{name:importlib.metadata.version(name) for name in ['torch','numpy','scikit-learn','scipy','joblib','matplotlib']})
        save_json(out/'invocation.json',{'argv':sys.argv,'cwd':str(Path.cwd()),'executable':sys.executable,'utc':datetime.now(timezone.utc).isoformat(),'fresh_training':True,'platform':platform.platform()})
        print('STUDY:',out,flush=True);print('CONFIG:',json.dumps(c,sort_keys=True),flush=True)
        make_data(source,out/'data',c)
    manifest=json.loads((out/'data/manifest.json').read_text())
    print('DATA:',json.dumps({'sources':manifest['sources'],'partitions':manifest['partitions'],'classes':manifest['class_counts'],'node_identity':manifest['node_identity'],'role':c['role']}),flush=True)
    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS=str(c['threads']),MKL_NUM_THREADS=str(c['threads']),LOKY_MAX_CPU_COUNT=str(c['threads']))
    for seed in c['seeds']:
        for model in MODELS:
            job=out/f'{model}_{seed}'
            if (job/'complete.json').exists():
                if not a.resume:raise ValueError('Unexpected existing result in a fresh run')
                verify_complete(job,manifest['fingerprint']);print(f'EXPLICIT RESUME: verified and reused completed {model} seed {seed}',flush=True);continue
            job.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u','-m','ids.theia.study_worker','--study',str(out),'--model',model,'--seed',str(seed)]
            if a.resume:cmd.append('--resume')
            with (job/'worker.log').open('a' if a.resume else 'w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                try:
                    for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    if proc.wait():raise RuntimeError(f'Worker failed: {job}. Resume explicitly with --resume {out}')
                except BaseException:
                    if proc.poll() is None:proc.terminate();proc.wait()
                    raise
    from .study_report import build
    build(out)
    print('Completed:',out/'dashboard.html',flush=True)


if __name__=='__main__':main()
