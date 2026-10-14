import argparse,csv,json,platform,random,resource,time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from ids.data import load
from ids.models import FlashInspired,MagicInspired
from ids.baselines import summary_features,shingle_features
from ids.metrics import threshold_from_validation,evaluate


def work(data,out,name,seed):
    out=Path(out); config=json.loads((out/'config.json').read_text())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config['threads'])
    # Tested CPU; explicit device prevents silent changes between models.
    if config['device']!='cpu': raise ValueError('This version supports CPU only for controlled comparison')
    torch.use_deterministic_algorithms(True)
    graphs,splits,manifest=load(data)
    train=splits['train']; val=splits['validation']; test=splits['test']
    benign=[i for i in train if graphs[i]['y']==0]
    y=lambda ids: np.array([graphs[i]['y'] for i in ids])
    history=[]; model=None; scaler=None; knn=None
    tensors=[(torch.from_numpy(g['x']),torch.from_numpy(g['src']),torch.from_numpy(g['dst'])) for g in graphs]
    start=time.perf_counter()
    if name in ['flash_inspired','magic_inspired']:
        dim=manifest['feature_dim']; hidden=config['hidden']
        model=FlashInspired(dim,hidden) if name=='flash_inspired' else MagicInspired(dim,hidden)
        optim=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        train_ids=train if name=='flash_inspired' else benign
        positive_weight=torch.tensor(sum(y(train)==0)/sum(y(train)==1),dtype=torch.float32)
        for epoch in range(config['epochs']):
            order=np.random.permutation(train_ids); loss_sum=0.; model.train()
            for i in order:
                x,s,d=tensors[i]; optim.zero_grad()
                if name=='flash_inspired':
                    label=torch.tensor(float(graphs[i]['y']))
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,s,d),label,pos_weight=positive_weight)
                else: loss=model.loss(x,s,d,config['mask_rate'])
                loss.backward(); optim.step(); loss_sum+=float(loss.detach())
            item={'epoch':epoch+1,'loss':loss_sum/len(order)}; history.append(item)
            print(f"epoch {epoch+1}/{config['epochs']} loss={item['loss']:.6f}",flush=True)
        model.eval()
        torch.save(model.state_dict(),out/f'{name}_{seed}.pt')
        if name=='magic_inspired':
            with torch.no_grad(): z=np.stack([model.embedding(*tensors[i]).numpy() for i in benign])
            scaler=StandardScaler().fit(z); z=scaler.transform(z)
            knn=NearestNeighbors(n_neighbors=min(config['knn_k'],len(z))).fit(z)
            np.savez_compressed(out/f'{name}_{seed}_reference.npz',mean=scaler.mean_,scale=scaler.scale_,z=z)
        def score(ids):
            with torch.no_grad():
                if name=='flash_inspired': return np.array([float(torch.sigmoid(model(*tensors[i]))) for i in ids])
                z=np.stack([model.embedding(*tensors[i]).numpy() for i in ids])
                return knn.kneighbors(scaler.transform(z),return_distance=True)[0].mean(1)
    elif name=='streamspot_inspired':
        z=np.stack([shingle_features(graphs[i]) for i in benign])
        knn=NearestNeighbors(n_neighbors=min(config['knn_k'],len(z))).fit(z)
        np.savez_compressed(out/f'{name}_{seed}_reference.npz',z=z)
        def score(ids):
            z=np.stack([shingle_features(graphs[i]) for i in ids])
            return knn.kneighbors(z,return_distance=True)[0].mean(1)
    elif name=='random_forest':
        x=np.stack([summary_features(graphs[i]) for i in train])
        forest=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=1)
        forest.fit(x,y(train))
        def score(ids):
            return forest.predict_proba(np.stack([summary_features(graphs[i]) for i in ids]))[:,1]
        # Checkpoint remains local trusted output, never deserialize uploaded input.
        import joblib
        joblib.dump(forest,out/f'{name}_{seed}.joblib')
    else: raise ValueError(name)
    training=time.perf_counter()-start
    begin=time.perf_counter(); val_scores=score(val); validation_seconds=time.perf_counter()-begin
    threshold=threshold_from_validation(y(val),val_scores)
    # Test partition evaluated once after threshold is fixed.
    begin=time.perf_counter(); test_scores=score(test); inference=time.perf_counter()-begin
    result=dict(model=name,seed=seed,**evaluate(y(test),test_scores,threshold),
                training_seconds=training,validation_seconds=validation_seconds,inference_seconds=inference,
                test_graphs=len(test),training_graphs=len(train) if name in ['flash_inspired','random_forest'] else len(benign),
                process_peak_rss_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if platform.system()=='Darwin' else 1024)),
                memory_definition='fresh worker process lifetime peak RSS; includes imports, data and model; CPU only',
                device='cpu',python_version=platform.python_version(),torch_version=str(torch.__version__),
                platform=platform.platform())
    (out/f'{name}_{seed}.json').write_text(json.dumps(result,indent=2))
    with open(out/f'{name}_{seed}_predictions.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['partition','graph_id','label','score','threshold','prediction'])
        writer.writeheader()
        for partition,ids,scores in [('validation',val,val_scores),('test',test,test_scores)]:
            for i,s in zip(ids,scores): writer.writerow(dict(partition=partition,graph_id=i,label=graphs[i]['y'],score=float(s),threshold=threshold,prediction=int(s>=threshold)))
    (out/f'{name}_{seed}_training.json').write_text(json.dumps(history,indent=2))
    print(f"test F1={result['f1']:.4f}, AP={result['average_precision']:.4f}, FPR={result['false_positive_rate']:.4f}",flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model',required=True); p.add_argument('--seed',type=int,required=True)
    a=p.parse_args(); work(a.data,a.output,a.model,a.seed)
