"""StreamSpot-style presentation for saved entity-level results; no model refitting."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import windows


def build(out):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    metrics = json.loads((out/'metrics.json').read_text())
    data = Path((out/'data_path.txt').read_text())
    records, totals = [], {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    names = {'flash_node':'FLASH-inspired','magic_node':'MAGIC-inspired','random_forest_node':'Random Forest'}
    first_seed = metrics[0]['seed']
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    curves = 0
    for m in metrics:
        counts = [m.get(k) for k in ('tn','fp','fn','tp')]
        m['accuracy'] = (counts[0]+counts[3])/sum(counts) if all(v is not None for v in counts) and sum(counts) else None
        with open(out/f"{m['model']}_{m['seed']}_predictions.csv",newline='') as stream:
            rows = []
            for r in csv.DictReader(stream):
                if r['partition'] != 'test': continue
                r.update(window=int(r['window']),label=int(r['label']),score=float(r['score']),
                         threshold=float(r['threshold']) if r['threshold'] else None,
                         prediction=int(r['prediction']) if r['prediction'] else None,
                         model=m['model'],seed=m['seed'])
                rows.append(r)
        totals[f"{m['model']}/{m['seed']}"] = len(rows)
        if m['seed'] == first_seed:
            known = [r for r in rows if r['label'] >= 0]
            y = np.asarray([r['label'] for r in known]); scores = np.asarray([r['score'] for r in known])
            if len(np.unique(y)) == 2:
                p,rec,_ = precision_recall_curve(y,scores); fp,tp,_ = roc_curve(y,scores)
                axes[0].plot(rec,p,label=names[m['model']]); axes[1].plot(fp,tp,label=names[m['model']]); curves += 1
        rows.sort(key=lambda r:(-r['score'],r['uuid']))
        records.extend(rows[:1000])
    for ax,title,xlabel,ylabel in zip(axes,['Precision–recall','ROC'],['Recall','False-positive rate'],['Precision','True-positive rate']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.05)); ax.grid(alpha=.2)
        if curves: ax.legend(fontsize=8)
        else: ax.text(.5,.5,'Unavailable: two labelled classes required',ha='center',wrap=True)
    fig.suptitle(f"{manifest['dataset']} · test entities · seed {first_seed}",fontsize=10)
    fig.tight_layout(); fig.savefig(out/'curves.png',dpi=150); plt.close(fig)
    selected = [m for m in metrics if m['seed']==first_seed]
    fig,axes = plt.subplots(1,len(selected),figsize=(12,3.6),squeeze=False)
    for ax,m in zip(axes[0],selected):
        ax.set_title(names[m['model']],fontsize=10)
        if any(m.get(k) is None for k in ('tn','fp','fn','tp')):
            ax.text(.5,.5,'No labelled decisions',ha='center'); ax.axis('off'); continue
        cm=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(cm,cmap='Blues'); ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Negative','Positive'])
        ax.set(xlabel='Prediction',ylabel='Recorded entity label')
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
    fig.suptitle('Exact entity decisions · unknown labels excluded',fontsize=10)
    fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    x=np.arange(len(selected)); keys=['accuracy','precision','recall','f1']; colors=['#137f75','#4388bb','#d59b42','#7568a6']
    for i,(key,color) in enumerate(zip(keys,colors)):
        values=[m.get(key) for m in selected]
        bars=ax.bar(x+(i-1.5)*.19,[100*v if v is not None else 0 for v in values],.19,label=key.replace('_',' ').title(),color=color)
        ax.bar_label(bars,labels=[f'{v*100:.1f}%' if v is not None else 'N/A' for v in values],fontsize=8,padding=3)
    ax.set_xticks(x,[names[m['model']] for m in selected]);ax.set(ylabel='Percent',ylim=(0,118),title=f'Test performance · seed {first_seed}')
    ax.legend(loc='upper center',ncol=4,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'performance.png',dpi=150);plt.close(fig)
    needed={}
    for row in records:needed.setdefault(row['window'],set()).add(row['uuid'])
    context={}; edge_names={v:k for k,v in manifest['edge_vocabulary'].items()}
    for entry,g in windows(data,'test'):
        if entry['window'] not in needed:continue
        lookup={str(u):i for i,u in enumerate(g['uuids'])}
        for node in needed[entry['window']]:
            center=lookup[node];used={center};chosen=[];seen=set();frontier={center}
            for _ in range(2):
                next_frontier=set()
                for j,(a,b) in enumerate(zip(g['src'],g['dst'])):
                    a,b=int(a),int(b)
                    if j in seen or not ({a,b}&frontier) or len(used|{a,b})>28:continue
                    chosen.append(j);seen.add(j);next_frontier|={a,b}-used;used|={a,b}
                    if len(chosen)>=40:break
                frontier=next_frontier
                if not frontier or len(chosen)>=40:break
            context[node]={'nodes':[{'id':i,'uuid':str(g['uuids'][i]),'type':str(g['kind'][i]),'selected':i==center} for i in sorted(used)],
                'edges':[{'source':int(g['src'][j]),'destination':int(g['dst'][j]),'type':edge_names.get(int(g['edge_types'][j]),'UNKNOWN_IN_TRAINING'),'timestamp_ns':str(g['timestamps'][j])} for j in chosen],
                'window_nodes':len(g['x']),'window_edges':len(g['src'])}
    payload={'metrics':metrics,'rows':records,'context':context,'totals':totals,'manifest':manifest,'first_seed':first_seed}
    template=(Path(__file__).parent/'dashboard_template.html').read_text()
    (out/'dashboard.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,allow_nan=False).replace('<','\\u003c')))
