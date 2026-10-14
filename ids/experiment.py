"""Each model runs in a fresh process for comparable process peak RSS."""
import csv,json,os,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

MODELS=['flash_inspired','magic_inspired','streamspot_inspired','random_forest']

def run(root,data_dir,config):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target=Path(root)/'results'/f"{config['name']}_{stamp}"
    target.mkdir(parents=True)
    (target/'config.json').write_text(json.dumps(config,indent=2))
    (target/'data_path.txt').write_text(str(Path(data_dir).relative_to(root)))
    for name in ['manifest.json','splits.json','statistics.json']:
        (target/name).write_bytes((Path(data_dir)/name).read_bytes())
    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS=str(config['threads']),MKL_NUM_THREADS=str(config['threads']))
    for seed in config['seeds']:
        for model in MODELS:
            print(f'Running {model}, seed {seed}',flush=True)
            with open(target/f'{model}_{seed}.log','w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','ids.worker','--data',str(data_dir),
                    '--output',str(target),'--model',model,'--seed',str(seed)],cwd=root,env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in proc.stdout:
                    print(line,end='',flush=True); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f'{model} failed; see {log.name}')
    rows=[]
    for seed in config['seeds']:
        for model in MODELS:
            rows.append(json.loads((target/f'{model}_{seed}.json').read_text()))
    (target/'metrics.json').write_text(json.dumps(rows,indent=2))
    with open(target/'metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    numeric=['precision','recall','f1','average_precision','roc_auc','false_positive_rate',
             'training_seconds','inference_seconds','process_peak_rss_mb']
    summary={}
    for model in MODELS:
        subset=[r for r in rows if r['model']==model]
        summary[model]={k:{'mean':float(np.mean([r[k] for r in subset])),
            'std':float(np.std([r[k] for r in subset],ddof=1)) if len(subset)>1 else None} for k in numeric}
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    from ids.demo import build
    build(target,Path(data_dir))
    print('Finished:',target,flush=True)
