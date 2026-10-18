"""Data-derived figures and bounded real-graph excerpts for the HTML report."""
import base64
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

COLORS = ['#128878', '#dc8151', '#526cca', '#9c67ad']
NAMES = {'flash_node':'FLASH-inspired','flash_inspired':'FLASH-inspired','magic_node':'MAGIC-inspired','magic_inspired':'MAGIC-inspired','random_forest':'Random Forest','random_forest_node':'Random Forest','streamspot_inspired':'StreamSpot-inspired'}


def make_visuals(runs, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#40585d','text.color':'#213e45','axes.edgecolor':'#dce5e4','grid.color':'#e8eeec','figure.facecolor':'white','axes.titleweight':'bold'})
    images = {}
    def save(fig, name):
        fig.savefig(destination/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(destination/f'{name}.svg',bbox_inches='tight')
        data=(destination/f'{name}.png').read_bytes()
        images[name]='data:image/png;base64,'+base64.b64encode(data).decode()
        plt.close(fig)
    for run in runs:
        seed = run['config']['seeds'][0]
        rows = [r for r in run['metrics'] if r['seed']==seed]
        fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
        for j,row in enumerate(rows):
            labels=[];scores=[]
            with (run['path']/f"{row['model']}_{seed}_predictions.csv").open() as f:
                for r in csv.DictReader(f):
                    if r['partition']=='test' and int(r['label']) in (0,1):
                        labels.append(int(r['label']));scores.append(float(r['score']))
            y=np.asarray(labels);s=np.asarray(scores)
            if len(y)!=sum(row[k] for k in ['tn','fp','fn','tp']):raise ValueError('Prediction coverage mismatch')
            if len(np.unique(y))<2:continue
            if abs(roc_auc_score(y,s)-row['roc_auc'])>1e-9 or abs(average_precision_score(y,s)-row['average_precision'])>1e-9:
                raise ValueError('Curve metrics disagree with saved results')
            fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
            axes[0].plot(fpr,tpr,color=COLORS[j],lw=2,label=f"{NAMES[row['model']]} · {row['roc_auc']:.3f}")
            axes[1].step(recall,precision,where='post',color=COLORS[j],lw=2,label=f"{NAMES[row['model']]} · {row['average_precision']:.3f}")
        axes[0].plot([0,1],[0,1],':',color='#9caeac',lw=1)
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve · AUC in legend')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall · AP in legend')
        for ax in axes:
            ax.set_xlim(-.015,1.015);ax.set_ylim(-.015,1.035);ax.grid(alpha=.7);ax.legend(loc='lower right' if ax==axes[0] else 'lower left',fontsize=8,frameon=True,facecolor='white')
        save(fig,run['dataset']+'_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax,run in zip(axes,runs):
        for j,row in enumerate([r for r in run['metrics'] if r['seed']==run['config']['seeds'][0] and ('flash' in r['model'] or 'magic' in r['model'])]):
            hist=json.loads((run['path']/f"{row['model']}_{row['seed']}_training.json").read_text())
            if isinstance(hist,dict):hist=hist['epochs']
            ax.plot([h['epoch'] for h in hist],[h.get('loss',h.get('mean_window_loss')) for h in hist],color=COLORS[j],lw=2,marker='o',markersize=3,label=NAMES[row['model']])
        ax.set(title='THEIA' if run['dataset']=='theia' else 'StreamSpot',xlabel='Epoch',ylabel='Training loss (log scale)',yscale='log');ax.grid(alpha=.7);ax.legend(fontsize=9)
    save(fig,'training_loss')
    return images


def graph_previews(runs):
    result={}
    s=runs[0];d=json.loads((s['path']/'demo_data.json').read_text());examples=[];seen=set()
    for label in [1,0]:
        for p in d['predictions']:
            key=str(p['graph_id'])
            if p['partition']=='test' and p['label']==label and key in d['evidence'] and key not in seen:
                seen.add(key);g=d['evidence'][key]
                examples.append(dict(title=f"Graph {key} · {'attack-labelled' if label else 'benign-labelled'}",nodes=g['nodes'],edges=g['edges'],total_nodes=g['sampled_nodes'],total_edges=g['sampled_edges'],note='Excerpt of an edge-sampled test graph. The label belongs to the whole graph; individual nodes have no attack labels.'))
                break
    result['streamspot']=examples
    t=runs[1];data=Path((t['path']/'data_path.txt').read_text().strip());entry=next(x for x in t['manifest']['shards'] if x['partition']=='test')
    with np.load(data/entry['file'],allow_pickle=False) as g:
        src=g['src'];dst=g['dst'];y=g['y'];types=g['types'];n=len(y)
        degree=np.bincount(np.concatenate([src,dst]),minlength=n)
        examples=[]
        for label in [1,0]:
            candidates=np.flatnonzero(y==label)
            if not len(candidates):continue
            center=int(candidates[np.argmax(degree[candidates])]);incident=np.flatnonzero((src==center)|(dst==center))
            nodes=[center]
            for i in incident:
                for node in [int(src[i]),int(dst[i])]:
                    if node not in nodes and len(nodes)<28:nodes.append(node)
            edges=[];selected=set(nodes)
            for i in incident:
                if int(src[i]) in selected and int(dst[i]) in selected:
                    edges.append(dict(source=int(src[i]),destination=int(dst[i]),type='stored directed edge'))
                    if len(edges)==40:break
            examples.append(dict(title=f"Node {center:,} · {'upstream positive' if label else 'presumed background'}",nodes=[dict(id=v,raw_id=str(g['uuids'][v]),type=f'type {types[v]}',label=int(y[v]),focus=v==center) for v in nodes],edges=edges,total_nodes=n,total_edges=len(src),note='Highest-degree node in this label group; at most 28 nodes / 40 incident edges. A selected illustration, not a representative sample or proof of an attack path.'))
        result['theia']=examples
    return result
