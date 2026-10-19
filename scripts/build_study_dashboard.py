"""Build the offline study explorer from existing saved results; never trains models."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from study_dashboard_data import enrich

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/comparison/dashboard.html'

def read(path):
    return json.loads(path.read_text())

def build():
    runs = []
    paths = sorted((ROOT / 'results').glob('*/metrics.json')) + sorted((ROOT / 'results/theia').glob('*/metrics.json'))
    paths += [d/'metrics.json' for d in sorted((ROOT/'results/imported').glob('*')) if d.is_dir()]
    paths += [d / 'metrics.json' for d in sorted((ROOT / 'results/theia').glob('study_*')) if d.is_dir() and not (d / 'metrics.json').exists()]
    for p in paths:
        rows = read(p) if p.exists() else [read(f) for f in sorted(p.parent.glob('*/metrics.json'))]
        if not isinstance(rows, list):
            continue
        folder = p.parent
        theia = folder.parent.name == 'theia' or (folder/'data/manifest.json').exists()
        name = ('imported/' if folder.parent.name=='imported' else '') + folder.name
        synthetic = 'SYNTHETIC' in name
        smoke = 'smoke' in name.lower()
        study = (folder/'data/manifest.json').exists() or folder.name.startswith('study_')
        status = 'Synthetic check' if synthetic else 'Smoke test' if smoke else 'Full study' if study else 'Historical run'
        if study and not (folder / 'study_complete.json').exists():
            status = 'Incomplete / failed audit'
        # Model rows must have internally consistent confusion counts.
        for row in rows:
            count = sum(row[k] for k in ('tn','fp','fn','tp'))
            expected = row.get('known_targets', row.get('test_graphs',count))
            if count != expected:
                raise ValueError(f'Invalid confusion counts: {p}')
        def link(file):
            return os.path.relpath(file, OUT.parent)
        artifacts = {str(f.relative_to(folder)):link(f) for f in folder.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.html','.log') and 'source_snapshot' not in f.parts and f.name not in ('demo_data.json',)}
        figs = list((folder/'figures').glob('*.png')) if (folder/'figures').exists() else list(folder.glob('*.png'))
        evidence = enrich(folder, rows)
        runs.append(dict(evidence=evidence,id=name,dataset='THEIA' if theia else 'StreamSpot',unit='nodes' if theia else 'graphs',status=status,metrics=rows,artifacts=artifacts,figures=[{'name':f.stem.replace('_',' ').title(),'url':link(f)} for f in sorted(figs)],config=read(folder/'config.json') if (folder/'config.json').exists() else {},audit=read(folder/'evaluation_audit.json') if (folder/'evaluation_audit.json').exists() else None))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    (OUT.parent/'study_index.json').write_text(json.dumps(dict(schema_version=1,runs=runs),allow_nan=False,indent=2))
    payload = json.dumps(dict(runs=runs,built=datetime.now(timezone.utc).isoformat()),allow_nan=False).replace('<','\\u003c')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text((ROOT/'ids/study_explorer_template.html').read_text().replace('__STUDY_DATA__',payload))
    print(f'{OUT}: {len(runs)} experiments, {sum(len(r["metrics"]) for r in runs)} saved model/seed results')

if __name__ == '__main__':
    build()
