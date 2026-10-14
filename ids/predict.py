"""Reload a trusted generated checkpoint and execute an actual forward pass."""
import json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from ids.data import load
from ids.models import FlashInspired,MagicInspired

def predict(root,run_dir,model_name,seed,graph_id):
    target=Path(root)/run_dir
    config=json.loads((target/'config.json').read_text())
    torch.set_num_threads(config['threads'])
    graphs,splits,manifest=load(Path(root)/(target/'data_path.txt').read_text().strip())
    if not 0<=graph_id<len(graphs): raise ValueError('Graph ID must be 0..599')
    model=(FlashInspired if model_name=='flash_inspired' else MagicInspired)(manifest['feature_dim'],config['hidden'])
    model.load_state_dict(torch.load(target/f'{model_name}_{seed}.pt',map_location='cpu',weights_only=True)); model.eval()
    g=graphs[graph_id]; inputs=[torch.from_numpy(g[k]) for k in ['x','src','dst']]
    start=time.perf_counter()
    with torch.no_grad():
        if model_name=='flash_inspired': score=float(torch.sigmoid(model(*inputs)))
        else:
            z=model.embedding(*inputs).numpy()[None,:]
            with np.load(target/f'{model_name}_{seed}_reference.npz',allow_pickle=False) as reference:
                z-=reference['mean']; z/=reference['scale']
                nn=NearestNeighbors(n_neighbors=min(config['knn_k'],len(reference['z']))).fit(reference['z'])
                score=float(nn.kneighbors(z)[0].mean())
    elapsed=time.perf_counter()-start
    metric=json.loads((target/f'{model_name}_{seed}.json').read_text())
    result={'model':model_name,'seed':seed,'graph_id':graph_id,
            'partition':next(k for k,v in splits.items() if graph_id in v),'score':score,
            'threshold':metric['threshold'],'alert':score>=metric['threshold'],
            'graph_ground_truth':g['y'],'forward_and_scoring_seconds':elapsed,
            'note':'Graph score, not individual node attribution. CPU checkpoint forward pass.'}
    print(json.dumps(result,indent=2)); return result
