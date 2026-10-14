import argparse,json,sys
from pathlib import Path
from ids.data import download,prepare
ROOT=Path(__file__).resolve().parent

def main():
    if '--dataset' in sys.argv or any(a.startswith('--dataset=') for a in sys.argv):
        probe=argparse.ArgumentParser(add_help=False); probe.add_argument('--dataset',choices=['streamspot','theia'])
        choice,remaining=probe.parse_known_args()
        if choice.dataset=='theia':
            from ids.theia.cli import main as theia_main
            return theia_main(ROOT,remaining)
        sys.argv=[sys.argv[0]]+remaining
    p=argparse.ArgumentParser(description='Controlled StreamSpot graph experiment')
    p.add_argument('command',choices=['download','prepare','run','demo','predict'])
    p.add_argument('--config',default='configs/quick.json')
    p.add_argument('--model',choices=['flash_inspired','magic_inspired'],default='flash_inspired')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--graph-id',type=int,default=300)
    p.add_argument('--run-dir'); p.add_argument('--port',type=int,default=8765)
    a=p.parse_args(); config=json.loads((ROOT/a.config).read_text())
    if a.command=='download': download(ROOT)
    elif a.command=='prepare': prepare(ROOT,config)
    elif a.command=='run':
        from ids.experiment import run
        run(ROOT,prepare(ROOT,config),config)
    elif a.command=='predict':
        from ids.predict import predict
        if not a.run_dir: p.error('predict requires --run-dir results/YOUR_RUN_FOLDER')
        predict(ROOT,a.run_dir,a.model,a.seed,a.graph_id)
    else:
        from ids.demo import serve
        if a.run_dir: target=ROOT/a.run_dir
        else:
            choices=sorted((ROOT/'results').glob('*/dashboard.html'),key=lambda p:p.stat().st_mtime)
            if not choices: p.error('No results: run python main.py run first')
            target=choices[-1].parent
        serve(target,a.port)
if __name__=='__main__': main()
