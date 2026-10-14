"""Isolated sequential workers and offline node-level result inspection."""
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .common import save_json, sha256
from .data import validate, windows
from .metrics import evaluate, select_threshold

MODELS = ('flash_node','magic_node','random_forest_node')


def run(root, data, config):
    root = Path(root); data = Path(data).resolve(); validate(data)
    manifest = json.loads((data/'manifest.json').read_text())
    for part in ('train','validation','test'):
        if not manifest['partitions'][part].get('targets',0):
            raise ValueError(f'No {part} targets: enlarge the input scope or predeclare suitable timestamp boundaries')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root/'results'/'theia'/f"{config['name']}_{stamp}"; out.mkdir(parents=True)
    save_json(out/'config.json',config)
    for name in ('manifest.json','splits.json','targets.csv'): shutil.copyfile(data/name,out/name)
    (out/'data_path.txt').write_text(str(data))
    # Capture code identity alongside data identity, including this implementation.
    code = list((root/'ids'/'theia').glob('*.py'))+[root/'ids'/'models.py',root/'ids'/'metrics.py',root/'main.py']
    save_json(out/'code_fingerprints.json',{str(p.relative_to(root)):sha256(p) for p in code})
    env = dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS=str(config['threads']),MKL_NUM_THREADS=str(config['threads']))
    for seed in config['seeds']:
        for model in MODELS:
            print(f'THEIA {model} seed {seed}',flush=True)
            with open(out/f'{model}_{seed}.log','w') as log:
                proc = subprocess.Popen([sys.executable,'-u','-m','ids.theia.worker','--data',str(data),
                    '--output',str(out),'--model',model,'--seed',str(seed)],cwd=root,env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in proc.stdout:
                    print(line,end='',flush=True); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f'Worker failed; see {log.name}')
    audit = reevaluate(out); build_dashboard(out)
    print('THEIA results:',out,flush=True)
    return out


def reevaluate(out):
    """Recompute saved metrics and assert every method used identical targets/labels."""
    out = Path(out); config = json.loads((out/'config.json').read_text()); metrics = []; expected = None
    for seed in config['seeds']:
        for model in MODELS:
            with open(out/f'{model}_{seed}_predictions.csv',newline='') as stream: rows = list(csv.DictReader(stream))
            identity = [(r['uuid'],r['partition'],r['window'],r['cutoff_ns'],r['label']) for r in rows]
            if expected is None: expected = identity
            if identity != expected: raise ValueError('Models have different evaluation targets or labels')
            val = [r for r in rows if r['partition']=='validation']; test = [r for r in rows if r['partition']=='test']
            threshold,rule = select_threshold([int(r['label']) for r in val],[float(r['score']) for r in val],config['validation_fpr'])
            frozen = json.loads((out/f'{model}_{seed}_threshold.json').read_text())
            if threshold != frozen['threshold']: raise ValueError('Threshold is not the validation-only selection')
            computed = evaluate([int(r['label']) for r in test],[float(r['score']) for r in test],threshold)
            saved = json.loads((out/f'{model}_{seed}.json').read_text())
            for key,value in computed.items():
                if value != saved[key]: raise ValueError(f'Saved metric mismatch: {model} {key}')
            for row in rows:
                decision = '' if threshold is None else str(int(float(row['score'])>=threshold))
                if row['prediction'] != decision: raise ValueError('Prediction/threshold mismatch')
            metrics.append(saved)
    save_json(out/'metrics.json',metrics)
    with open(out/'metrics.csv','w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    audit = {'shared_targets':True,'validation_only_thresholds':True,'metrics_recomputed':True,
             'target_rows_per_model':len(expected),'models':list(MODELS)}
    save_json(out/'evaluation_audit.json',audit)
    return audit


def build_dashboard(out):
    from .dashboard import build
    build(out)


def verify_checkpoints(out):
    """Actual forward passes from saved checkpoints against every saved target."""
    from .worker import load_scorer,score_graph
    import torch
    out = Path(out); config = json.loads((out/'config.json').read_text())
    torch.set_num_threads(config['threads']); data = Path((out/'data_path.txt').read_text()); validate(data)
    checked = 0; max_error = 0.
    for seed in config['seeds']:
        for name in MODELS:
            model,scaler,reference = load_scorer(out,name,seed)
            with open(out/f'{name}_{seed}_predictions.csv',newline='') as stream:
                expected = {r['uuid']:float(r['score']) for r in csv.DictReader(stream)}
            seen = set()
            for part in ('validation','test'):
                for entry,g in windows(data,part):
                    scores = score_graph(name,model,g,scaler,reference)
                    for i in np.flatnonzero(g['target']):
                        node = str(g['uuids'][i]); err = abs(float(scores[i])-expected[node]); max_error = max(max_error,err)
                        if err > 1e-6: raise ValueError(f'Checkpoint score mismatch: {name} {node}: {err}')
                        seen.add(node); checked += 1
            if seen != expected.keys(): raise ValueError('Checkpoint target coverage mismatch')
    audit = {'checked_scores':checked,'max_absolute_error':max_error,'tolerance':1e-6}
    save_json(out/'checkpoint_audit.json',audit); return audit
