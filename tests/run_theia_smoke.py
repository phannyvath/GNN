"""Reproducible engineering execution on SYNTHETIC CDM18, never a benchmark."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from theia_fixture import make_fixture
from ids.theia.common import save_json
from ids.theia.data import prepare
from ids.theia.experiment import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',choices=['smoke','pilot'],default='smoke')
    p.add_argument('--output',required=True,help='New fixture directory (must not exist)')
    a=p.parse_args();directory=Path(a.output)
    if directory.exists():p.error('Use a new output directory')
    config=json.loads((ROOT/f'configs/theia_{a.size}.json').read_text())
    config.update(dataset='synthetic_cdm18',name=f'SYNTHETIC_{a.size}_NOT_BENCHMARK',
        split_mode='explicit',start_ns=0,train_end_ns=1000,validation_end_ns=2000,end_ns=3000,
        window_ns=1000,experiment_role=f'synthetic_{a.size}')
    raw,labels=make_fixture(directory,24 if a.size=='smoke' else 10000)
    save_json(directory/'config.json',config)
    data=prepare([raw],directory/'processed',config,curated_path=labels)
    run(ROOT,data,config)
