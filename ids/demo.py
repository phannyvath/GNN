"""Offline results browser. Displays recorded experiments, not a live network IDS."""
import csv,json,functools,http.server
from pathlib import Path
import numpy as np
from ids.data import load


def build(target,data_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve,roc_curve
    target=Path(target); graphs,splits,manifest=load(data_dir)
    metrics=json.loads((target/'metrics.json').read_text()); predictions=[]
    for m in metrics:
        with open(target/f"{m['model']}_{m['seed']}_predictions.csv") as f:
            for r in csv.DictReader(f):
                predictions.append({k:(int(v) if k in ['graph_id','label','prediction'] else float(v) if k in ['score','threshold'] else v) for k,v in r.items()}|{'model':m['model'],'seed':m['seed']})
    first_seed=metrics[0]['seed']; fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for m in [r for r in metrics if r['seed']==first_seed]:
        rows=[r for r in predictions if r['seed']==first_seed and r['model']==m['model'] and r['partition']=='test']
        y=[r['label'] for r in rows]; s=[r['score'] for r in rows]
        p,rec,_=precision_recall_curve(y,s); fp,tp,_=roc_curve(y,s)
        axes[0].plot(rec,p,label=m['model'].replace('_',' ')); axes[1].plot(fp,tp,label=m['model'].replace('_',' '))
    axes[0].axhline(1/6,color='grey',ls='--',label='test attack prevalence')
    axes[0].set(xlabel='Recall',ylabel='Precision',title=f'Test precision–recall curves · seed {first_seed}',xlim=(0,1),ylim=(0,1.05))
    axes[1].plot([0,1],[0,1],'--',color='grey'); axes[1].set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Test ROC curves · seed {first_seed}',xlim=(0,1),ylim=(0,1.05))
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(target/'curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5))
    for ax,m in zip(axes,[r for r in metrics if r['seed']==first_seed]):
        cm=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(cm,cmap='Blues'); ax.set_title(m['model'].replace('_',' '),fontsize=9)
        ax.set_xticks([0,1],['Benign','Attack']); ax.set_yticks([0,1],['Benign','Attack'])
        ax.set_xlabel('Prediction'); ax.set_ylabel('Actual graph label')
        for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(target/'confusion_matrices.png',dpi=160); plt.close(fig)
    evidence={}
    inverse_node={str(v):k for k,v in manifest['node_type_vocabulary'].items()}
    inverse_edge={str(v):k for k,v in manifest['edge_type_vocabulary'].items()}
    for i in splits['test']:
        g=graphs[i]; edges=[]; used=set()
        for a,b,r,o in zip(g['src'],g['dst'],g['edge_types'],g['edge_ordinals']):
            if len(used|{int(a),int(b)})>28: continue
            used|={int(a),int(b)}; edges.append({'source':int(a),'destination':int(b),'type':inverse_edge.get(str(r),'unknown'),'ordinal':o})
            if len(edges)>=40: break
        evidence[str(i)]={'nodes':[{'id':j,'raw_id':g['node_ids'][j],'type':inverse_node.get(str(g['types'][j]),'unknown')} for j in sorted(used)],
            'edges':edges,'sampled_nodes':len(g['x']),'sampled_edges':len(g['src']),'raw_edges':g['raw_edges']}
    payload={'metrics':metrics,'predictions':predictions,'manifest':manifest,'splits':splits,'evidence':evidence}
    (target/'demo_data.json').write_text(json.dumps(payload))
    template=(Path(__file__).parent/'dashboard_template.html').read_text()
    (target/'dashboard.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload).replace('<','\\u003c')))


def serve(target,port):
    target=Path(target)
    if not (target/'dashboard.html').exists(): raise FileNotFoundError('Missing dashboard.html')
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(target))
    print(f'Open http://127.0.0.1:{port}/dashboard.html — Ctrl+C to stop',flush=True)
    with http.server.ThreadingHTTPServer(('127.0.0.1',port),handler) as server: server.serve_forever()
