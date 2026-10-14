"""Called by main.py --dataset theia; StreamSpot's default CLI is preserved."""
import argparse
import json
from pathlib import Path
from .common import save_json


def main(root, argv):
    p = argparse.ArgumentParser(description='THEIA node-level experiment (independent of StreamSpot)')
    p.add_argument('command',choices=['download','prepare','validate','run','evaluate','demo'])
    p.add_argument('--dataset',choices=['theia'],default='theia')
    p.add_argument('--config',default='configs/theia_smoke.json')
    p.add_argument('--input',nargs='+',help='CDM18 JSONL, gzip JSONL or JSON tar.gz archives')
    p.add_argument('--data-dir',default='data/theia/processed/smoke')
    p.add_argument('--output',default='data/theia/raw')
    p.add_argument('--labels',default='docs/theia_sources/theia_malicious.txt')
    p.add_argument('--curated-labels',help='CSV with uuid,label,evidence; absence means unknown')
    p.add_argument('--archive',choices=['1r','6r'],default='1r')
    p.add_argument('--sample-records',type=int,default=0)
    p.add_argument('--max-download-mib',type=int,default=2048)
    p.add_argument('--rescore',action='store_true',help='Re-evaluate locally produced checkpoints against all saved scores')
    p.add_argument('--run-dir'); p.add_argument('--port',type=int,default=8766)
    a = p.parse_args(argv); root = Path(root)
    if a.command == 'download':
        from .download import download
        if a.sample_records < 0 or a.max_download_mib <= 0: p.error('Download limits must be positive')
        return download(root/a.output,a.archive,a.sample_records,a.max_download_mib)
    if a.command in ('prepare','run'):
        config = json.loads((root/a.config).read_text())
        required_positive = ('max_raw_records','max_import_nodes','max_import_edges','max_line_bytes','max_edges_per_window',
            'max_nodes_per_window','max_windows_per_partition','window_ns','epochs','hidden','threads',
            'rf_max_train_nodes','max_reference_nodes','knn_k','rf_trees','rf_max_depth')
        if any(config[k]<=0 for k in required_positive) or config['max_nodes_per_window']<2:
            p.error('All resource limits must be positive; windows require at least two nodes')
        if not 0 < config['mask_rate'] <= 1 or not 0 < config['validation_fpr'] < 1: p.error('Invalid mask/FPR rates')
    if a.command == 'prepare':
        from .data import prepare
        if not a.input: p.error('prepare requires --input FILE [FILE ...]')
        return prepare([root/x for x in a.input],root/a.data_dir,config,root/a.labels,root/a.curated_labels if a.curated_labels else None)
    if a.command == 'validate':
        from .data import validate
        print(json.dumps(validate(root/a.data_dir),indent=2)); return
    if a.command == 'run':
        from .experiment import run
        return run(root,root/a.data_dir,config)
    if not a.run_dir: p.error('evaluate/demo requires --run-dir results/theia/YOUR_RUN')
    if a.command == 'evaluate':
        from .experiment import reevaluate,build_dashboard
        print(reevaluate(root/a.run_dir)); build_dashboard(root/a.run_dir)
        if a.rescore:
            from .experiment import verify_checkpoints
            print(verify_checkpoints(root/a.run_dir))
    else:
        from ids.demo import serve
        serve(root/a.run_dir,a.port)
