"""Fresh, isolated study workers; exact all-node reference and explicit epoch resume."""
import argparse
import csv
import hashlib
import json
import os
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

from .common import fingerprint, peak_rss_mib, save_json, sha256
from .metrics import evaluate, select_threshold
from .models import FlashNode, MagicNode
from .study_data import batches


def emit(event, **fields):
    print(json.dumps({'event':event,**fields},allow_nan=False),flush=True)


def atomic_torch(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def finite(x):
    if not np.isfinite(np.asarray(x)).all():raise ValueError('Nonfinite model output')
    return x


def args_for(g,scaler):
    return (torch.from_numpy(scaler.transform(g['x']).astype(np.float32)),torch.from_numpy(g['src']),torch.from_numpy(g['dst']))


def objective(model,name,args,g,rate):
    x,src,dst=args
    if name=='flash_node':
        mask=torch.from_numpy(g['types']>0)
        if not mask.any():raise ValueError('No known types for validation/training objective')
        logits=model(x,src,dst)
        loss=torch.nn.functional.cross_entropy(logits[mask],torch.from_numpy(g['types'])[mask])
        return loss,int(mask.sum()),float((logits[mask].argmax(1)==torch.from_numpy(g['types'])[mask]).float().mean())
    mask=torch.rand(len(x))<rate
    if not mask.any():mask[0]=True
    damaged=x.clone();damaged[mask]=model.token
    output=model.decoder(model.encode(damaged,src,dst))
    loss=torch.nn.functional.mse_loss(output[mask],x[mask])
    return loss,int(mask.sum()),None


class ExactReference:
    """Deduplicate exact embeddings but preserve kNN multiplicity of ALL occurrences."""
    def __init__(self,z,counts,k,batch=256,threads=4):
        finite(z)
        if len(z)!=len(counts) or not len(z) or (counts<=0).any():raise ValueError('Invalid reference counts')
        self.z=z;self.counts=counts;self.k=min(k,int(counts.sum()));self.batch=batch
        self.index=NearestNeighbors(n_neighbors=min(self.k,len(z)),algorithm='kd_tree',n_jobs=threads).fit(z)

    def score(self,z):
        finite(z);result=[]
        for start in range(0,len(z),self.batch):
            dist,idx=self.index.kneighbors(z[start:start+self.batch]);mult=self.counts[idx]
            before=np.cumsum(mult,axis=1)-mult
            take=np.minimum(mult,np.maximum(0,self.k-before))
            result.append((dist*take).sum(1)/self.k)
        return finite(np.concatenate(result))


def fit_reference(model,scaler,data,c,out):
    m=json.loads((data/'manifest.json').read_text());n=m['partitions']['train']['targets']
    z=np.lib.format.open_memmap(out/'reference_all.npy',mode='w+',dtype=np.float32,shape=(n,c['hidden']))
    offset=0;model.eval()
    with (out/'reference_targets.csv').open('w',newline='') as stream,torch.no_grad():
        writer=csv.writer(stream);writer.writerow(['row','graph_local_id'])
        for entry,g in batches(data,'train',c):
            block=model.encode(*args_for(g,scaler)).numpy();finite(block)
            z[offset:offset+len(block)]=block
            for i,uid in enumerate(g['uuids']):writer.writerow([offset+i,uid])
            offset+=len(block);emit('reference_build',nodes_processed=offset,total_nodes=n)
    if offset!=n:raise ValueError('Reference omitted training nodes')
    z.flush();unique,inverse,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    np.savez_compressed(out/'reference.npz',z=unique,counts=counts,inverse=inverse)
    del z
    (out/'reference_all.npy').unlink()
    emit('reference_ready',all_training_occurrences=n,unique_embeddings=len(unique),sampling='none; multiplicity preserved exactly')
    return ExactReference(unique,counts,c['knn_k'],c['query_batch_nodes'],c['threads']),n,len(unique)


def score(name,model,g,scaler,reference):
    with torch.no_grad():
        if name=='random_forest_node':
            probabilities=model.predict_proba(scaler.transform(g['x']));scores=np.ones(len(g['x']))
            for j,klass in enumerate(model.classes_):
                mask=g['types']==klass;scores[mask]-=probabilities[mask,j]
        elif name=='flash_node':
            probs=model(*args_for(g,scaler)).softmax(1).numpy()
            scores=1-probs[np.arange(len(probs)),g['types']];scores[g['types']==0]=1
        else:scores=reference.score(model.encode(*args_for(g,scaler)).numpy())
    return finite(scores)


def inference(name,model,scaler,reference,data,c,part,out,threshold=None):
    started=time.perf_counter();score_seconds=0.;ys=[];ss=[];identity=hashlib.sha256();count=0
    with (out/f'{part}_predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['graph_local_id','partition','label','score','threshold','prediction'])
        for entry,g in batches(data,part,c):
            t=time.perf_counter();values=score(name,model,g,scaler,reference);score_seconds+=time.perf_counter()-t
            ys.append(g['y']);ss.append(values)
            for uid,label,value in zip(g['uuids'],g['y'],values):
                identity.update(f'{uid}\t{part}\t{label}\n'.encode())
                writer.writerow([uid,part,int(label),float(value),'' if threshold is None else threshold,'' if threshold is None else int(value>=threshold)])
            count+=len(values);emit('inference',partition=part,nodes_processed=count,elapsed_seconds=time.perf_counter()-started)
    # CSV values are Python float64. Select thresholds in the same precision so
    # nextafter tie handling is exactly reproducible when predictions are read.
    return np.concatenate(ys),np.concatenate(ss).astype(np.float64),{'scoring_seconds':score_seconds,'wall_seconds_including_csv':time.perf_counter()-started,'target_identity_sha256':identity.hexdigest()}


def run(study,name,seed,resume=False):
    study=Path(study);data=study/'data';out=study/f'{name}_{seed}';out.mkdir(exist_ok=True)
    c=json.loads((study/'config.json').read_text());m=json.loads((data/'manifest.json').read_text())
    torch.set_num_threads(c['threads']);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.use_deterministic_algorithms(True)
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'device':'cpu','threads':c['threads'],'seed':seed,
         'memory_definition':'Fresh worker process OS high-water RSS measured after predictions; includes imports, model selection, training, reference and scoring. MiB; excludes other processes. On epoch resume each attempt has its own peak.'}
    save_json(out/'environment.json',env)
    emit('start',model=name,environment=env,config=c,dataset_fingerprint=m['fingerprint'],counts=m['partitions'],class_counts=m['class_counts'],source_sha256=m['sources'],checkpoint_reuse='explicit epoch resume if available' if resume else 'none; fresh initialization')
    started=time.perf_counter();scaler=StandardScaler()
    for entry,g in batches(data,'train',c):scaler.partial_fit(g['x'])
    n=int(scaler.n_samples_seen_);dim=m['feature_dim'];ntypes=len(m['type_vocabulary'])+1
    save_json(out/'preprocessing.json',{'fit_partition':'train','node_occurrences':n,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'malicious_label_use':'none'})
    history=[];reference=None;selected_epoch=None;ref_n=0;ref_unique=0;previous_seconds=0.;previous_peak=0.
    if name=='random_forest_node':
        x=np.lib.format.open_memmap(out/'rf_inputs.npy',mode='w+',dtype=np.float32,shape=(n,dim));y=np.lib.format.open_memmap(out/'rf_types.npy',mode='w+',dtype=np.int64,shape=(n,));offset=0
        for entry,g in batches(data,'train',c):
            if (g['types']==0).any():raise ValueError('Unknown training type')
            size=len(g['x']);x[offset:offset+size]=scaler.transform(g['x']);y[offset:offset+size]=g['types'];offset+=size
            emit('rf_input_batch',nodes_processed=offset,total_nodes=n,sampling='none')
        if offset!=n:raise ValueError('RF omitted training nodes')
        x.flush();y.flush()
        model=RandomForestClassifier(n_estimators=c['rf_trees'],max_depth=c['rf_max_depth'],min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=c['threads'],warm_start=True)
        # Growing trees with a fixed RNG seed preserves sklearn's fresh-forest sequence.
        for trees in range(min(10,c['rf_trees']),c['rf_trees']+10,10):
            trees=min(trees,c['rf_trees']);model.set_params(n_estimators=trees);model.fit(x,y)
            elapsed=time.perf_counter()-started
            emit('rf_fit',trees_completed=trees,total_trees=c['rf_trees'],eligible_nodes_per_tree=n,bootstrap=True,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/trees*(c['rf_trees']-trees))
            if trees==c['rf_trees']:break
        joblib.dump({'model':model,'scaler':scaler,'fingerprint':m['fingerprint']},out/'checkpoint.joblib')
        del x,y;(out/'rf_inputs.npy').unlink();(out/'rf_types.npy').unlink()
        selection='Fixed prespecified forest; no epoch selection; threshold fitted on validation only.'
    else:
        model=FlashNode(dim,c['hidden'],ntypes) if name=='flash_node' else MagicNode(dim,c['hidden'])
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
        best=float('inf');stale=0;start_epoch=0
        if resume and (out/'last.pt').exists():
            state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            if state['fingerprint']!=m['fingerprint'] or state['config_fingerprint']!=fingerprint(c):raise ValueError('Resume checkpoint mismatch')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
            best=state['best'];selected_epoch=state['best_epoch'];stale=state['stale'];start_epoch=state['epoch'];history=state['history'];previous_seconds=state['elapsed_seconds'];previous_peak=state['peak_rss_mib']
            emit('resume_epoch',completed_epoch=start_epoch,best_epoch=selected_epoch)
        else:
            atomic_torch(out/'initial.pt',{'model':model.state_dict(),'seed':seed})
            emit('fresh_weights',initial_checkpoint_sha256=sha256(out/'initial.pt'))
        total_batches=len(FILES_TRAIN:= [e for e in m['shards'] if e['partition']=='train'])
        for epoch in range(start_epoch,c['epochs']):
            if stale>=c['patience']:break
            epoch_started=time.perf_counter();model.train();weighted=0.;target_count=0;nodes=0;edges=0
            for batch,(entry,g) in enumerate(batches(data,'train',c),1):
                optimizer.zero_grad();loss,targets,acc=objective(model,name,args_for(g,scaler),g,c['mask_rate'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradients')
                optimizer.step();weighted+=float(loss.detach())*targets;target_count+=targets;nodes+=len(g['x']);edges+=len(g['src'])
                elapsed=time.perf_counter()-epoch_started
                emit('training_batch',model=name,seed=seed,epoch=epoch+1,batches_completed=batch,total_batches=total_batches,nodes_processed=nodes,edges_processed=edges,loss_targets=target_count,loss=float(loss.detach()),elapsed_seconds=previous_seconds+time.perf_counter()-started,estimated_epoch_remaining_seconds=elapsed/batch*(total_batches-batch))
                del loss
            model.eval();val_weighted=0.;val_targets=0;val_acc=0.;val_started=time.perf_counter()
            with torch.no_grad(),torch.random.fork_rng():
                torch.manual_seed(c['validation_mask_seed'])
                for entry,g in batches(data,'validation',c):
                    loss,targets,acc=objective(model,name,args_for(g,scaler),g,c['mask_rate']);finite(float(loss))
                    val_weighted+=float(loss)*targets;val_targets+=targets
                    if acc is not None:val_acc+=acc*targets
            val_loss=val_weighted/val_targets;selected=val_loss<best-c['min_delta']
            if selected:
                best=val_loss;selected_epoch=epoch+1;stale=0
                atomic_torch(out/'best.pt',{'model':model.state_dict(),'epoch':selected_epoch,'validation_loss':best,'fingerprint':m['fingerprint'],'dim':dim,'types':ntypes,'seed':seed})
            else:stale+=1
            row={'epoch':epoch+1,'batches':total_batches,'nodes_processed':nodes,'edges_processed':edges,'loss_targets':target_count,'train_loss':weighted/target_count,'validation_loss':val_loss,'validation_objective_targets':val_targets,'validation_type_accuracy':val_acc/val_targets if name=='flash_node' else None,'selected':selected,'selected_epoch':selected_epoch,'reason':'lowest validation objective so far' if selected else 'validation objective did not improve by min_delta','seconds':time.perf_counter()-epoch_started,'selection_validation_seconds':time.perf_counter()-val_started}
            history.append(row);save_json(out/'epochs.json',history)
            elapsed=previous_seconds+time.perf_counter()-started
            atomic_torch(out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'epoch':epoch+1,'best':best,'best_epoch':selected_epoch,'stale':stale,'history':history,'fingerprint':m['fingerprint'],'config_fingerprint':fingerprint(c),'elapsed_seconds':elapsed,'peak_rss_mib':max(previous_peak,peak_rss_mib())})
            emit('epoch_complete',**row,elapsed_seconds=elapsed,estimated_remaining_training_seconds=float(np.mean([r['seconds'] for r in history]))*(c['epochs']-epoch-1),eta_note='epoch budget only; early stopping can shorten it; excludes reference/scoring')
        state=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);model.eval()
        selection='Minimum validation node-type CE' if name=='flash_node' else 'Minimum validation masked reconstruction MSE with fixed validation mask'
        joblib.dump(scaler,out/'scaler.joblib')
        if name=='magic_node':reference,ref_n,ref_unique=fit_reference(model,scaler,data,c,out)
    training_seconds=previous_seconds+time.perf_counter()-started
    save_json(out/'selection.json',{'selected_epoch':selected_epoch,'criterion':selection,'test_labels_used':False,'history_epochs':len(history)})
    yv,sv,val_t=inference(name,model,scaler,reference,data,c,'validation',out)
    threshold,rule=select_threshold(yv,sv,c['validation_fpr'])
    save_json(out/'threshold.json',{'threshold':threshold,'rule':rule,'partition':'validation'})
    vm=evaluate(yv,sv,threshold);save_json(out/'validation_metrics.json',vm);emit('validation_selected_checkpoint',metrics=vm,selection=selection,epoch=selected_epoch,threshold=threshold,rule=rule)
    yt,st,test_t=inference(name,model,scaler,reference,data,c,'test',out,threshold)
    # Recompute metrics from the actual saved CSV, not transient arrays.
    yy=[];ss=[]
    with (out/'test_predictions.csv').open() as stream:
        for row in csv.DictReader(stream):yy.append(int(row['label']));ss.append(float(row['score']))
    metrics=evaluate(yy,ss,threshold)
    metrics.update(model=name,seed=seed,role=c['role'],dataset=m['dataset'],fingerprint=m['fingerprint'],selected_epoch=selected_epoch,
        training_seconds=training_seconds,validation_seconds=val_t['wall_seconds_including_csv'],inference_seconds=test_t['scoring_seconds'],test_export_wall_seconds=test_t['wall_seconds_including_csv'],process_peak_rss_mib=max(previous_peak,peak_rss_mib()),
        training_node_occurrences=n,training_edges_per_epoch=m['partitions']['train']['edges'],epochs_completed=len(history),neural_training_node_visits=sum(r['nodes_processed'] for r in history),rf_fit_nodes=n if name=='random_forest_node' else None,
        reference_occurrences=ref_n,reference_unique_embeddings=ref_unique,validation_targets=len(yv),test_targets=len(yt),threshold=threshold,
        training_time_definition='Scaler, optimization, epoch-validation, checkpointing and reference construction; excludes process startup and shared data preparation.',inference_time_definition='Test model forward and scoring only; excludes graph reads, CSV writes and metric computation.',device='cpu')
    save_json(out/'metrics.json',metrics)
    artifact_names=['metrics.json','test_predictions.csv','validation_predictions.csv','threshold.json','selection.json','preprocessing.json']
    artifact_names+=['checkpoint.joblib'] if name=='random_forest_node' else ['best.pt','last.pt','initial.pt','scaler.joblib','epochs.json']
    if name=='magic_node':artifact_names+=['reference.npz','reference_targets.csv']
    save_json(out/'complete.json',{'fingerprint':m['fingerprint'],'test_identity':test_t['target_identity_sha256'],'validation_identity':val_t['target_identity_sha256'],'artifacts':{f:sha256(out/f) for f in artifact_names}})
    emit('complete',**metrics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--model',required=True,choices=['flash_node','magic_node','random_forest_node']);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.study,a.model,a.seed,a.resume)


if __name__=='__main__':main()
