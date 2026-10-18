"""Build an offline HTML comparison from saved experiment artifacts."""
import argparse
import json
import os
from comparison_visuals import make_visuals, graph_previews
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_run(path, dataset):
    path = path.resolve()
    def read(name):
        return json.loads((path/name).read_text())
    metrics, config, manifest = read('metrics.json'), read('config.json'), read('manifest.json')
    if dataset == 'theia' and (manifest.get('smoke_nodes') is not None or manifest.get('format_version') != 'processed_static_v1'):
        raise ValueError('Choose a complete preprocessed THEIA run, not a smoke/synthetic run')
    expected = {'flash_node','magic_node','random_forest_node'} if dataset == 'theia' else {'flash_inspired','magic_inspired','random_forest','streamspot_inspired'}
    if {r['model'] for r in metrics} != expected:
        raise ValueError(f'Incomplete models: {path}')
    for row in metrics:
        total = sum(row[k] for k in ['tn','fp','fn','tp'])
        if total != row.get('known_targets', row.get('test_graphs')):
            raise ValueError('Confusion counts do not match test targets')
        row['memory_mib'] = row.get('process_peak_rss_mib',row.get('process_peak_rss_mb'))
        row['dataset_key'] = dataset
    splits = read('splits.json')
    return dict(name=path.name, dataset=dataset, metrics=metrics, config=config, manifest=manifest, splits=splits, path=path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--streamspot', default='results/quick_20261013T063017021814Z')
    p.add_argument('--theia', default='results/theia/processed_full_20261018T132637258404Z')
    p.add_argument('--output', default='results/comparison/dashboard.html')
    a = p.parse_args(); output = ROOT/a.output
    runs = [load_run(ROOT/a.streamspot,'streamspot'),load_run(ROOT/a.theia,'theia')]
    images = make_visuals(runs, output.parent/'figures')
    previews = graph_previews(runs)
    for run in runs:
        run['previews'] = previews[run['dataset']]
        run['images'] = {k:v for k,v in images.items() if k.startswith(run['dataset']) or k == 'training_loss'}
        path = run.pop('path')
        run['links'] = {name:Path(os.path.relpath(path/name,output.parent)).as_posix() for name in ['metrics.csv','manifest.json','config.json']}
    payload = json.dumps(runs,allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    template = (ROOT/'ids/comparison_template.html').read_text()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__RUN_DATA__',payload))
    print(output)


if __name__ == '__main__':
    main()
