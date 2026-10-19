"""Audit saved predictions, aggregate seeds and plot data-derived study evidence."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from .common import save_json
from .metrics import evaluate, select_threshold

MODELS=('flash_node','magic_node','random_forest_node')
NAMES={'flash_node':'FLASH-inspired','magic_node':'MAGIC-inspired','random_forest_node':'Random Forest'}
METRICS=('precision','recall','f1','average_precision','roc_auc','false_positive_rate','tn','fp','fn','tp','training_seconds','inference_seconds','process_peak_rss_mib')


def load_predictions(path,partition,threshold=None):
    y=[];scores=[];ids=[];identity=hashlib.sha256()
    with Path(path).open() as stream:
        for r in csv.DictReader(stream):
            label=int(r['label']);score=float(r['score'])
            if r['partition']!=partition or label not in (-1,0,1) or not np.isfinite(score):raise ValueError('Invalid saved prediction')
            if threshold is not None and (float(r['threshold'])!=threshold or int(r['prediction'])!=int(score>=threshold)):raise ValueError('Prediction disagrees with frozen threshold')
            identity.update(f"{r['graph_local_id']}\t{partition}\t{label}\n".encode());y.append(label);scores.append(score);ids.append(r['graph_local_id'])
    return np.asarray(y),np.asarray(scores),ids,identity.hexdigest()


def aggregate(rows):
    output=[]
    for model in MODELS:
        group=[r for r in rows if r['model']==model]
        for metric in METRICS:
            values=[r[metric] for r in group if r[metric] is not None]
            output.append({'model':model,'metric':metric,'mean':float(np.mean(values)) if values else None,'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,'defined_seeds':len(values),'total_seeds':len(group)})
    return output


def verify_saved_checkpoints(study):
    import joblib
    import torch
    from .models import FlashNode, MagicNode
    from .study_worker import ExactReference, score
    from .study_data import batches
    study=Path(study);c=json.loads((study/'config.json').read_text())
    torch.set_num_threads(c['threads'])
    checked=0;maximum=0.
    for seed in c['seeds']:
        for name in MODELS:
            out=study/f'{name}_{seed}';reference=None
            if name=='random_forest_node':
                state=joblib.load(out/'checkpoint.joblib');model=state['model'];scaler=state['scaler']
            else:
                state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
                model=FlashNode(state['dim'],c['hidden'],state['types']) if name=='flash_node' else MagicNode(state['dim'],c['hidden'])
                model.load_state_dict(state['model']);model.eval();scaler=joblib.load(out/'scaler.joblib')
                if name=='magic_node':
                    with np.load(out/'reference.npz',allow_pickle=False) as ref:
                        reference=ExactReference(ref['z'],ref['counts'],c['knn_k'],c['query_batch_nodes'],c['threads'])
            for part in ['validation','test']:
                _,expected,ids,_=load_predictions(out/f'{part}_predictions.csv',part);offset=0
                for entry,g in batches(study/'data',part,c):
                    actual=score(name,model,g,scaler,reference);end=offset+len(actual)
                    if ids[offset:end]!=g['uuids'].tolist():raise ValueError('Checkpoint target order mismatch')
                    err=float(np.max(np.abs(actual-expected[offset:end])))
                    if err>1e-6:raise ValueError('Saved checkpoint does not reproduce predictions')
                    maximum=max(maximum,err);checked+=len(actual);offset=end
                if offset!=len(expected):raise ValueError('Checkpoint coverage mismatch')
    result={'checked_scores':checked,'max_absolute_error':maximum,'tolerance':1e-6,'checkpoints_reloaded':True}
    save_json(study/'checkpoint_audit.json',result)
    return result


def audit(study):
    from .study import verify_complete
    from .study_data import batches
    c=json.loads((study/'config.json').read_text());m=json.loads((study/'data/manifest.json').read_text());expected={}
    for part in ['validation','test']:
        digest=hashlib.sha256()
        for entry,g in batches(study/'data',part,c):
            for uid,label in zip(g['uuids'],g['y']):digest.update(f'{uid}\t{part}\t{label}\n'.encode())
        expected[part]=digest.hexdigest()
    rows=[];examples={};curve_data={}
    for seed in c['seeds']:
        for name in MODELS:
            out=study/f'{name}_{seed}';verify_complete(out,m['fingerprint'])
            yv,sv,_,iv=load_predictions(out/'validation_predictions.csv','validation')
            threshold,rule=select_threshold(yv,sv,c['validation_fpr']);frozen=json.loads((out/'threshold.json').read_text())
            if frozen!={'threshold':threshold,'rule':rule,'partition':'validation'}:raise ValueError('Threshold is not validation-only')
            yt,st,ids,it=load_predictions(out/'test_predictions.csv','test',threshold)
            if iv!=expected['validation'] or it!=expected['test']:raise ValueError('Evaluation target/label mismatch')
            recomputed=evaluate(yt,st,threshold);saved=json.loads((out/'metrics.json').read_text())
            for key,value in recomputed.items():
                if saved[key]!=value:raise ValueError(f'Metric mismatch: {name}/{seed}/{key}')
            val=evaluate(yv,sv,threshold)
            if val!=json.loads((out/'validation_metrics.json').read_text()):raise ValueError('Validation metrics mismatch')
            rows.append(saved)
            known=yt>=0;curve_data[(name,seed)]=(yt[known],st[known])
            if seed==c['seeds'][0]:
                pred=st>=threshold if threshold is not None else np.full(len(st),False)
                examples[name]={}
                for label,mask in [('correct_detection',(yt==1)&pred),('false_positive',(yt==0)&pred),('missed_attack',(yt==1)&~pred)]:
                    found=np.flatnonzero(mask);examples[name][label]=None if not len(found) or threshold is None else {'graph_local_id':ids[int(found[0])],'score':float(st[int(found[0])]),'label':int(yt[int(found[0])]),'threshold':threshold}
    save_json(study/'metrics.json',rows)
    with (study/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=aggregate(rows);save_json(study/'summary.json',summary)
    with (study/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save_json(study/'evaluation_audit.json',{'all_saved_metrics_recomputed':True,'thresholds_validation_only':True,'identical_targets_and_labels_all_methods_and_seeds':True,'expected_target_hashes':expected,'workers':len(rows),'seeds':c['seeds']})
    return c,m,rows,summary,curve_data,examples


def plot_figures(study,c,m,rows,summary,curves,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','axes.labelcolor':'#405865','text.color':'#213e45'})
    directory=study/'figures';directory.mkdir(exist_ok=True);plots=[]
    def save(fig,name):
        fig.savefig(directory/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(directory/f'{name}.svg',bbox_inches='tight');plt.close(fig);plots.append(name)
    colors=['#168775','#d7814f','#526dc5'];seeds=c['seeds']
    fig,axes=plt.subplots(2,len(seeds),figsize=(4.2*len(seeds),7),squeeze=False,layout='constrained')
    for j,seed in enumerate(seeds):
        for i,name in enumerate(MODELS):
            y,s=curves[(name,seed)]
            if len(np.unique(y))!=2:continue
            fpr,tpr,_=roc_curve(y,s);p,r,_=precision_recall_curve(y,s)
            axes[0,j].plot(fpr,tpr,color=colors[i],label=NAMES[name],lw=1.7)
            axes[1,j].step(r,p,where='post',color=colors[i],label=NAMES[name],lw=1.7)
        axes[0,j].plot([0,1],[0,1],':',color='#b8c5c9');axes[0,j].set(title=f'ROC · seed {seed}',xlabel='False positive rate',ylabel='True positive rate')
        axes[1,j].set(title=f'Precision–recall · seed {seed}',xlabel='Recall',ylabel='Precision')
        for ax in axes[:,j]:
            ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2);ax.legend(fontsize=7)
            if len(np.unique(curves[(MODELS[0],seed)][0]))<2:ax.text(.5,.5,'Undefined: one labelled class',ha='center')
    save(fig,'roc_pr')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,name in enumerate(MODELS[:2]):
        for j,seed in enumerate(seeds):
            h=json.loads((study/f'{name}_{seed}/epochs.json').read_text());epochs=[r['epoch'] for r in h]
            axes[i,0].plot(epochs,[r['train_loss'] for r in h],color=colors[j%3],label=f'seed {seed}')
            axes[i,1].plot(epochs,[r['validation_loss'] for r in h],color=colors[j%3],label=f'seed {seed}')
            selected=json.loads((study/f'{name}_{seed}/selection.json').read_text())['selected_epoch'];r=h[selected-1]
            axes[i,1].scatter([selected],[r['validation_loss']],color=colors[j%3],marker='*',s=75)
        for j,kind in enumerate(['Training','Validation']):
            axes[i,j].set(title=f'{NAMES[name]} · {kind}',xlabel='Epoch',ylabel='Objective loss (log)',yscale='log');axes[i,j].legend();axes[i,j].grid(alpha=.2)
    save(fig,'learning_curves')
    fig,axes=plt.subplots(3,len(seeds),figsize=(4*len(seeds),9),squeeze=False,layout='constrained')
    for i,name in enumerate(MODELS):
        for j,seed in enumerate(seeds):
            r=next(r for r in rows if r['model']==name and r['seed']==seed);ax=axes[i,j]
            if r['tn'] is None:ax.text(.5,.5,'No validation threshold',ha='center');continue
            counts=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]]);rates=counts/np.maximum(1,counts.sum(1,keepdims=True))
            ax.imshow(rates,cmap='Blues',vmin=0,vmax=1)
            for a in range(2):
                for b in range(2):ax.text(b,a,f'{counts[a,b]:,}\n{rates[a,b]:.1%}',ha='center',va='center',color='white' if rates[a,b]>.5 else '#203d4a')
            ax.set(xticks=[0,1],xticklabels=['Negative','Positive'],yticks=[0,1],yticklabels=['Background','Positive'],xlabel='Predicted',ylabel='Actual / presumed',title=f'{NAMES[name]} · seed {seed}')
    save(fig,'confusion_matrices')
    keys=['precision','recall','f1','average_precision','roc_auc','false_positive_rate','training_seconds','inference_seconds','process_peak_rss_mib']
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for ax,key in zip(axes.flat,keys):
        for i,name in enumerate(MODELS):
            r=next(r for r in summary if r['model']==name and r['metric']==key)
            if r['mean'] is not None:ax.bar(i,r['mean'],yerr=r['sample_std'] or 0,color=colors[i],capsize=4,width=.6)
        ax.set(title=key.replace('_',' '),xticks=[0,1,2],xticklabels=['FLASH','MAGIC','RF']);ax.grid(axis='y',alpha=.2)
        if key in keys[:6]:ax.set_ylim(0,1.08)
    save(fig,'model_comparison')
    entry=next(e for e in m['shards'] if e['partition']=='test')
    with np.load(study/'data'/entry['file'],allow_pickle=False) as g:src=g['src'];dst=g['dst'];labels=g['y'];ids=g['uuids']
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained');records=[]
    for i,name in enumerate(MODELS):
        for j,category in enumerate(['correct_detection','false_positive','missed_attack']):
            ax=axes[i,j];sample=examples[name][category];ax.axis('off');ax.set_title(f'{NAMES[name]} · {category.replace("_"," ")}',fontsize=9)
            if sample is None:ax.text(.5,.5,'No such outcome in this run',ha='center',transform=ax.transAxes);continue
            center=int(sample['graph_local_id'].rsplit(':',1)[1]);indices=np.flatnonzero((src==center)|(dst==center));selected=[center]
            for idx in indices:
                for v in [int(src[idx]),int(dst[idx])]:
                    if v not in selected and len(selected)<20:selected.append(v)
            edges=[]
            for idx in indices:
                if int(src[idx]) in selected and int(dst[idx]) in selected:
                    edges.append([int(src[idx]),int(dst[idx])])
                    if len(edges)>=30:break
            coords={center:(0.,0.)}
            for k,node in enumerate(selected[1:]):coords[node]=(float(np.cos(2*np.pi*k/max(1,len(selected)-1))),float(np.sin(2*np.pi*k/max(1,len(selected)-1))))
            for a,b in edges:
                ax.annotate('',xy=coords[b],xytext=coords[a],arrowprops={'arrowstyle':'->','color':'#b3c5cb','lw':.6,'shrinkA':5,'shrinkB':5})
            for node in selected:
                x,y=coords[node];ax.scatter([x],[y],s=100 if node==center else 40,c='#d1784c' if labels[node]==1 else '#168775',zorder=3)
                if node==center:ax.text(x,y-.18,str(node),fontsize=7,ha='center')
            ax.set_xlim(-1.3,1.3);ax.set_ylim(-1.3,1.3);ax.text(.5,-.04,f"score {sample['score']:.4g} · threshold {sample['threshold']:.4g}",ha='center',transform=ax.transAxes,fontsize=7)
            records.append({'model':name,'seed':seeds[0],'category':category,**sample,'node_ids':[str(ids[v]) for v in selected],'edges':edges,'node_labels':[int(labels[v]) for v in selected]})
    save(fig,'graph_context');save_json(study/'graph_context.json',{'selection':'First graph-local index in each outcome category, first seed only; at most 20 nodes/30 incident edges','interpretation':'Observed graph context, not a proven model explanation or causal attack path. Orange: upstream positive; teal: presumed background. Scores are not attack probabilities.','excerpts':records})
    return plots


def build(study):
    study=Path(study);c,m,rows,summary,curves,examples=audit(study)
    if c['smoke_nodes'] is not None:verify_saved_checkpoints(study)
    plots=plot_figures(study,c,m,rows,summary,curves,examples)
    def esc(v):return html.escape(str(v))
    heads=['Model','Seed','Selected epoch','Precision','Recall','F1','AP','ROC-AUC','FPR','Train seconds','Test scoring seconds','Peak MiB']
    table=[]
    for r in rows:
        values=[NAMES[r['model']],r['seed'],r['selected_epoch']]+[r[k] for k in ['precision','recall','f1','average_precision','roc_auc','false_positive_rate','training_seconds','inference_seconds','process_peak_rss_mib']]
        table.append('<tr>'+''.join('<td>'+('—' if v is None else esc(f'{v:.4f}' if isinstance(v,float) else v))+'</td>' for v in values)+'</tr>')
    descriptions={'roc_pr':'Full labelled test predictions, one panel per seed. AP is average precision, not trapezoidal PR area.',
        'learning_curves':'Training and validation self-supervised objectives; stars mark selected epochs. Different objectives are not comparable losses. No test labels select epochs.',
        'confusion_matrices':'Counts and row-normalised rates using frozen validation thresholds. Background labels are presumed.',
        'model_comparison':'Per-model mean with sample standard deviation (ddof=1) across configured seeds. Seed variability is not a confidence interval over attacks.',
        'graph_context':'Observed neighbourhoods for first-seed outcomes; selected excerpts, not explanations. Orange = upstream positive; teal = presumed background. Missing outcomes are explicitly marked.'}
    figures=[]
    for name in plots:
        blob=base64.b64encode((study/'figures'/f'{name}.png').read_bytes()).decode()
        figures.append(f'<section><h2>{esc(name.replace("_"," ").title())}</h2><p>{esc(descriptions[name])}</p><img src="data:image/png;base64,{blob}" alt="{esc(name)}"><p><a href="figures/{name}.svg" download>Download SVG</a> · <a href="figures/{name}.png" download>Download PNG</a></p></section>')
    counts=''.join(f'<li>{esc(part)}: {v["targets"]:,} node occurrences / {v["edges"]:,} edges; labels {esc(m["class_counts"][part])}</li>' for part,v in m['partitions'].items())
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>THEIA reproducible study</title><style>body{{font:14px/1.6 system-ui;background:#eef3f5;color:#203b49;margin:0}}main{{max-width:1250px;margin:auto;padding:32px}}header{{background:#173b45;color:white;padding:28px;border-radius:12px}}h1{{margin:0}}section{{background:white;padding:24px;border:1px solid #dbe6e9;border-radius:10px;margin:24px 0}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}td,th{{padding:11px;border-bottom:1px solid #dce7e9;text-align:left}}a{{color:#087d6a}}.notice{{padding:16px;background:#fff4df;border-left:3px solid #bd8542}}code{{overflow-wrap:anywhere}}details pre{{white-space:pre-wrap}}@media(max-width:700px){{main{{padding:14px}}section{{padding:15px}}}}</style><main><header><h1>THEIA · Reproducible experiment</h1><p>{esc(c['role'])} · seeds {esc(c['seeds'])} · CPU</p></header><p class="notice">{'SMOKE TEST ONLY — induced node-prefix subset; not full-study performance.' if c['smoke_nodes'] else 'Full exported training split, not all nodes in the dataset. Validation and test graphs remain held out.'} Graph-local identities do not establish unique entity disjointness. Presumed-background labels and prior test-set inspection limit benchmark claims.</p><section><h2>Data and protocol</h2><ul>{counts}</ul><p>All eligible training nodes used; no benign-only filtering. RF fits all training rows using standard bootstrap trees. MAGIC retains all training embeddings with exact multiplicity compression. One complete graph per neural batch within declared node/edge caps.</p><p>Training time includes scaling, optimization, epoch validation, checkpoints and reference construction. Test scoring time excludes graph reads and CSV writing. Peak memory is worker OS high-water RSS, not whole-machine memory.</p><p><a href="metrics.csv">Per-seed metrics CSV</a> · <a href="summary.csv">Mean / sample SD CSV</a> · <a href="evaluation_audit.json">Prediction audit</a> · <a href="data/alignment_audit.json">Label alignment audit</a> · <a href="graph_context.json">Graph excerpt records</a></p><details><summary>Exact configuration and data fingerprint</summary><pre>{esc(json.dumps(c,indent=2))}</pre><code>{esc(m['fingerprint'])}</code></details></section><section><h2>Measured per-seed test results</h2><div class="scroll"><table><thead><tr>{''.join('<th>'+h+'</th>' for h in heads)}</tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Undefined metrics are —. Scores are uncalibrated anomaly scores, not probabilities of attack.</p></section>{''.join(figures)}<footer>Generated from saved predictions and logs. StreamSpot remains a separate experiment; its graph-level metrics are not pooled here.</footer></main></html>'''
    (study/'dashboard.html').write_text(doc)
    save_json(study/'study_complete.json',{'role':c['role'],'seeds':c['seeds'],'models':list(MODELS),'evaluations':len(rows),'figures':plots,'data_fingerprint':m['fingerprint']})
    return study/'dashboard.html'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--rescore',action='store_true');a=p.parse_args();print(build(Path(a.run)))
    if a.rescore:print(verify_saved_checkpoints(Path(a.run)))


if __name__=='__main__':main()
