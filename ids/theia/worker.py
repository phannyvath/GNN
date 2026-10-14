"""One model per process, one graph shard at a time; train labels never consumed."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .common import peak_rss_mib, save_json
from .data import windows
from .metrics import evaluate, select_threshold
from .models import FlashNode, MagicNode


def reservoir(rows, maximum, rng):
    selected = []; seen = 0
    for row in rows:
        seen += 1
        if len(selected) < maximum: selected.append(row)
        else:
            j = int(rng.integers(seen))
            if j < maximum: selected[j] = row
    if not selected: raise ValueError('No eligible training nodes; increase/import an appropriate chronological scope')
    return selected, seen


def tensors(g, scaler):
    x = torch.from_numpy(scaler.transform(g['x']).astype(np.float32))
    return x, torch.from_numpy(g['src']), torch.from_numpy(g['dst'])


def score_graph(model_name, model, graph, scaler, reference=None):
    with torch.no_grad():
        if model_name == 'random_forest_node':
            proba = model.predict_proba(scaler.transform(graph['x']))
            scores = np.ones(len(graph['x']),dtype=np.float64)
            for i, klass in enumerate(model.classes_):
                match = graph['types'] == klass; scores[match] -= proba[match,i]
            return scores
        args = tensors(graph,scaler)
        if model_name == 'flash_node':
            proba = model(*args).softmax(1).numpy()
            scores = 1 - proba[np.arange(len(proba)),graph['types']]
            scores[graph['types'] == 0] = 1
            return scores
        z = model.encode(*args).numpy()
        # Query in bounded chunks: do not allocate n_nodes x reference_size distances.
        return np.concatenate([reference.kneighbors(z[i:i+256])[0].mean(1) for i in range(0,len(z),256)])


def run(data, output, model_name, seed):
    data = Path(data); output = Path(output)
    config = json.loads((output/'config.json').read_text()); manifest = json.loads((data/'manifest.json').read_text())
    torch.set_num_threads(config['threads']); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    if config['device'] != 'cpu': raise ValueError('This validated implementation supports CPU only; MPS is not assumed compatible')
    start = time.perf_counter(); scaler = StandardScaler()
    count = 0
    for _,g in windows(data,'train'):
        scaler.partial_fit(g['x']); count += len(g['x'])
    if not count: raise ValueError('No training windows')
    dim = manifest['feature_dim']; types = len(manifest['type_vocabulary'])+1
    model = None; reference = None; training = []; reference_ids = []; train_seen = 0
    if model_name == 'random_forest_node':
        def rows():
            for _,g in windows(data,'train'):
                x = scaler.transform(g['x']).astype(np.float32)
                for i in np.flatnonzero(g['target'] & (g['types'] > 0)):
                    yield (x[i].copy(),int(g['types'][i]),str(g['uuids'][i]))
        sample,train_seen = reservoir(rows(),config['rf_max_train_nodes'],rng)
        x = np.stack([r[0] for r in sample]); y = np.asarray([r[1] for r in sample])
        reference_ids = [r[2] for r in sample]
        model = RandomForestClassifier(n_estimators=config['rf_trees'],max_depth=config['rf_max_depth'],
            min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=config['threads'])
        model.fit(x,y); del sample,x,y
        joblib.dump({'model':model,'scaler':scaler},output/f'{model_name}_{seed}.joblib')
    else:
        model = FlashNode(dim,config['hidden'],types) if model_name == 'flash_node' else MagicNode(dim,config['hidden'])
        optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        for epoch in range(config['epochs']):
            model.train(); losses = []; steps = 0
            for _,g in windows(data,'train'):
                x,src,dst = tensors(g,scaler); optimizer.zero_grad()
                if model_name == 'flash_node':
                    mask = torch.from_numpy(g['target'] & (g['types'] > 0))
                    if not mask.any(): continue
                    loss = torch.nn.functional.cross_entropy(model(x,src,dst)[mask],torch.from_numpy(g['types'])[mask])
                else: loss = model.loss(x,src,dst,config['mask_rate'])
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step(); losses.append(float(loss.detach())); steps += 1
            if not steps: raise ValueError('No trainable nodes')
            training.append({'epoch':epoch+1,'mean_window_loss':float(np.mean(losses)),'windows':steps})
            print(json.dumps(training[-1]),flush=True)
        model.eval()
        if model_name == 'magic_node':
            def embeddings():
                with torch.no_grad():
                    for _,g in windows(data,'train'):
                        z = model.encode(*tensors(g,scaler)).numpy()
                        for i in np.flatnonzero(g['target']): yield (z[i].copy(),str(g['uuids'][i]))
            sample,train_seen = reservoir(embeddings(),config['max_reference_nodes'],rng)
            z = np.stack([r[0] for r in sample]); reference_ids = [r[1] for r in sample]
            reference = NearestNeighbors(n_neighbors=min(config['knn_k'],len(z)),algorithm='brute',n_jobs=config['threads']).fit(z)
            np.savez_compressed(output/f'{model_name}_{seed}_reference.npz',z=z,uuids=np.asarray(reference_ids))
            del sample
        torch.save({'state_dict':model.state_dict(),'dim':dim,'hidden':config['hidden'],'types':types,
                    'scaler_mean':torch.tensor(scaler.mean_), 'scaler_scale':torch.tensor(scaler.scale_),
                    'dataset_fingerprint':manifest['fingerprint'],'model':model_name,'seed':seed},output/f'{model_name}_{seed}.pt')
    training_seconds = time.perf_counter()-start
    save_json(output/f'{model_name}_{seed}_training.json',{'epochs':training,'reservoir_seen':train_seen,
        'reservoir_retained':len(reference_ids),'reference_uuids':reference_ids,
        'scaler_samples':int(scaler.n_samples_seen_),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
        'label_use':'No malicious/benign labels read in training, scaling, type vocabulary or reservoir selection.'})
    def infer(part):
        result = []; started = time.perf_counter()
        for entry,g in windows(data,part):
            scores = score_graph(model_name,model,g,scaler,reference)
            for i in np.flatnonzero(g['target']):
                result.append({'uuid':str(g['uuids'][i]),'window':entry['window'],'partition':part,
                    'cutoff_ns':entry['max_ns'],'label':int(g['y'][i]),'score':float(scores[i])})
        return result,time.perf_counter()-started
    val,validation_seconds = infer('validation')
    threshold,rule = select_threshold([r['label'] for r in val],[r['score'] for r in val],config['validation_fpr'])
    # Freeze the threshold before opening test shards.
    save_json(output/f'{model_name}_{seed}_threshold.json',{'threshold':threshold,'rule':rule,'partition':'validation'})
    test,inference_seconds = infer('test')
    metrics = evaluate([r['label'] for r in test],[r['score'] for r in test],threshold)
    metrics.update(dataset=manifest['dataset'],target=manifest['target'],model=model_name,seed=seed,
        fingerprint=manifest['fingerprint'],label_policy=manifest['config']['label_policy'],threshold=threshold,threshold_rule=rule,
        training_seconds=training_seconds,validation_seconds=validation_seconds,inference_seconds=inference_seconds,
        process_peak_rss_mib=peak_rss_mib(),training_context_node_occurrences=count,
        test_targets=len(test),validation_targets=len(val),device='cpu',experiment_role=config.get('experiment_role','engineering_only'))
    with open(output/f'{model_name}_{seed}_predictions.csv','w',newline='') as stream:
        fields = ['uuid','window','partition','cutoff_ns','label','score','threshold','prediction']
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for r in val+test:
            writer.writerow(r|{'threshold':threshold,'prediction':None if threshold is None else int(r['score']>=threshold)})
    metrics['process_peak_rss_mib'] = peak_rss_mib()
    save_json(output/f'{model_name}_{seed}.json',metrics)
    save_json(output/f'{model_name}_{seed}_environment.json',{'platform':platform.platform(),'machine':platform.machine(),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,
        'threads':config['threads'],'memory_definition':'OS high-water resident set measured after prediction CSV writing in a fresh worker, MiB; includes imports, training and inference; excludes other processes and GPU memory.'})
    print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['flash_node','magic_node','random_forest_node'],required=True); p.add_argument('--seed',type=int,required=True)
    a = p.parse_args(); run(a.data,a.output,a.model,a.seed)


def load_scorer(output, model_name, seed):
    """Reload locally produced checkpoints; never load external pickle/joblib models."""
    output = Path(output); config = json.loads((output/'config.json').read_text())
    if model_name == 'random_forest_node':
        state = joblib.load(output/f'{model_name}_{seed}.joblib')
        return state['model'],state['scaler'],None
    state = torch.load(output/f'{model_name}_{seed}.pt',map_location='cpu',weights_only=True)
    scaler = StandardScaler(); scaler.mean_ = state['scaler_mean'].numpy(); scaler.scale_ = state['scaler_scale'].numpy()
    scaler.n_features_in_ = state['dim']
    model = FlashNode(state['dim'],state['hidden'],state['types']) if model_name == 'flash_node' else MagicNode(state['dim'],state['hidden'])
    model.load_state_dict(state['state_dict']); model.eval(); reference = None
    if model_name == 'magic_node':
        with np.load(output/f'{model_name}_{seed}_reference.npz',allow_pickle=False) as archive: z = archive['z']
        reference = NearestNeighbors(n_neighbors=min(config['knn_k'],len(z)),algorithm='brute',n_jobs=config['threads']).fit(z)
    return model,scaler,reference
