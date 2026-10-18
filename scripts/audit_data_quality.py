"""Read-only dataset audit. Outputs diagnostics, never training or model selection."""
import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ids.theia.processed import read_graph, validate_processed
from ids.theia.common import sha256, save_json


def distribution(a):
    values, counts = np.unique(a, return_counts=True)
    return {str(v):int(c) for v,c in zip(values,counts)}


def ranking(y, score):
    if len(np.unique(y)) != 2:return {'roc_auc':None,'average_precision':None}
    return {'roc_auc':float(roc_auc_score(y,score)), 'average_precision':float(average_precision_score(y,score))}


def structure(src, dst, n, edge_type=None):
    deg=np.bincount(np.concatenate([src,dst]),minlength=n)
    keys=np.column_stack([src,dst] if edge_type is None else [src,dst,edge_type])
    unique=len(np.unique(keys,axis=0))
    matrix=coo_matrix((np.ones(len(src),dtype=np.int8),(src,dst)),shape=(n,n)).tocsr()
    count, labels=connected_components(matrix,directed=True,connection='weak')
    sizes=np.bincount(labels)
    return {'nodes':n,'edges':len(src),'isolated_nodes':int((deg==0).sum()),
        'self_loops':int((src==dst).sum()),'repeated_edge_occurrences':int(len(src)-unique),
        'edge_key':'source,destination,type' if edge_type is not None else 'source,destination (edge types unavailable here)',
        'weak_components':int(count),'largest_component_nodes':int(sizes.max()),
        'degree_quantiles':{str(q):float(np.quantile(deg,q)) for q in [0,.5,.9,.99,1]}}, deg


def feature_rows(x):
    x=np.ascontiguousarray(x)
    return x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).ravel()


def prediction_subgroups(run, report):
    with np.load(run/'data/test_4.npz',allow_pickle=False) as g:
        types=g['types']-1; y=g['y']; src=g['src']; dst=g['dst']
    results={}
    for model in ['flash_node','magic_node','random_forest_node']:
        scores=np.full(len(y),np.nan); predictions=np.zeros(len(y),dtype=int)
        with (run/f'{model}_42_predictions.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['partition']!='test':continue
                i=int(row['uuid'].rsplit(':',1)[1])
                if int(row['label'])!=y[i] or np.isfinite(scores[i]):raise ValueError('Prediction alignment failure')
                scores[i]=float(row['score']);predictions[i]=int(row['prediction'])
        if not np.isfinite(scores).all():raise ValueError('Missing prediction scores')
        results[model]=[]
        for t in np.unique(types):
            mask=types==t; truth=y[mask]; pred=predictions[mask]
            results[model].append(dict(raw_type=int(t),nodes=int(mask.sum()),positives=int(truth.sum()),
                **ranking(truth,scores[mask]),tp=int(((truth==1)&(pred==1)).sum()),
                fp=int(((truth==0)&(pred==1)).sum()),fn=int(((truth==1)&(pred==0)).sum())))
    report['theia_model_results_by_raw_type']=results
    degree=np.bincount(np.concatenate([src,dst]),minlength=len(y));center=int(degree.argmax())
    neighbors=np.unique(np.concatenate([dst[src==center],src[dst==center]]))
    report['theia_highest_degree_test_node']=dict(graph_local_index=center,degree=int(degree[center]),
        label=int(y[center]),unique_neighbors=len(neighbors),positive_neighbors=int(y[neighbors].sum()),
        fraction_all_positives_adjacent_to_this_node=float(y[neighbors].sum()/y.sum()),
        interpretation='Diagnostic concentration of annotated positives around one hub; not proof of an attack path or leakage.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-upstream',action='store_true')
    a=p.parse_args();started=time.perf_counter()
    out=ROOT/'results/data_quality';out.mkdir(exist_ok=True)
    source=ROOT/'data/theia';run=ROOT/'results/theia/processed_full_20261018T132637258404Z';data=run/'data'
    m=json.loads((data/'manifest.json').read_text())
    report={'date':'2026-10-19','scope':'Existing THEIA exports and prepared StreamSpot data; diagnostic audit, not a new benchmark.',
        'limitations':['Graph-local IDs cannot establish entity disjointness across exports.',
        'No raw timestamps: no temporal ordering or duplicate-event verification.',
        'Background is presumed, not independently verified benign.',
        'Shortcut rankings use test labels for diagnostic reporting only; do not tune models against these observations. No direction is selected to maximise AUC.',
        'Repeated seeds and alternative splits were not run; this audit cannot establish performance stability.']}
    provenance={'local_archive_sha256':sha256(source/'graphs.zip')}
    prior=out/'audit.json'
    if prior.exists():
        old=json.loads(prior.read_text()).get('theia_provenance',{})
        if old.get('local_archive_sha256')==provenance['local_archive_sha256'] and old.get('exact_upstream_match'):
            provenance.update(old)
            provenance['upstream_check_reused']=not a.verify_upstream
    with zipfile.ZipFile(source/'graphs.zip') as z:
        provenance['zip_crc_passed']=z.testzip() is None
        provenance['extracted_files_match_archive']={name:hashlib.sha256(z.read(name)).hexdigest()==sha256(source/name) for name in z.namelist() if not name.endswith('/')}
    if a.verify_upstream:
        import urllib.request
        url='https://raw.githubusercontent.com/FDUDSDE/MAGIC/aa0b647eea74b6faa0e52eb444370c4411a32cbe/data/theia/graphs.zip'
        try:
            h=hashlib.sha256();size=0
            with urllib.request.urlopen(url,timeout=30) as stream:
                while block:=stream.read(1024*1024):h.update(block);size+=len(block)
            provenance.update(upstream_url=url,remote_bytes=size,remote_sha256=h.hexdigest(),exact_upstream_match=h.hexdigest()==provenance['local_archive_sha256'])
        except Exception as e:provenance.update(upstream_url=url,verification_error=str(e))
    report['theia_provenance']=provenance
    report['theia_prepared_integrity']=validate_processed(data)
    report['theia_sources_match_manifest']={name:sha256(source/name)==value for name,value in m['sources'].items()}
    meta=json.loads((source/'metadata.json').read_text())
    report['theia_label_metadata']={'positive_indices':len(meta['malicious'][0]),'unique_positive_indices':len(set(meta['malicious'][0])),
        'missing_associated_names':meta['malicious'][1].count('NA'),'note':'Associated names are not a complete UUID mapping.'}
    graph_stats=[];train_unique=[];validation_unique=None;test_payload=None
    for entry in m['shards']:
        name=m['counts'][entry['window']]['file']
        src,dst,nt,et,_,_=read_graph(source/name)
        stats,degree=structure(src,dst,len(nt),et)
        stats.update(file=name,partition=entry['partition'],node_types=distribution(nt),edge_types=distribution(et),schema_valid=True)
        with np.load(data/entry['file'],allow_pickle=False) as g:
            x=g['x'];y=g['y'];rows=feature_rows(x);unique, inverse, counts=np.unique(rows,return_inverse=True,return_counts=True)
            stats.update(feature_columns=x.shape[1],finite_features=bool(np.isfinite(x).all()),constant_feature_columns=np.flatnonzero(np.ptp(x,axis=0)==0).tolist(),
                unique_feature_rows=len(unique),nodes_in_repeated_feature_groups=int(counts[counts>1].sum()))
            if entry['partition']=='train':train_unique.append(unique.copy())
            elif entry['partition']=='validation':validation_unique=unique.copy()
            else:
                stats['labels']=distribution(y)
                positives=np.bincount(inverse,weights=(y==1).astype(int),minlength=len(unique)).astype(int)
                negatives=counts-positives;conflict=(positives>0)&(negatives>0)
                report['theia_feature_label_ambiguity']={'groups_with_both_labels':int(conflict.sum()),
                    'nodes_in_conflicting_groups':int(counts[conflict].sum()),
                    'minimum_errors_for_any_deterministic_feature_only_classifier':int(np.minimum(positives,negatives).sum()),
                    'note':'Identical feature rows can represent different entities. This is a feature-only limitation; graph context can distinguish some nodes.'}
                type_table=[]
                for typ in sorted(set(nt.tolist())):
                    mask=nt==typ;type_table.append({'raw_type':int(typ),'nodes':int(mask.sum()),'positive_nodes':int((y[mask]==1).sum()),'positive_rate':float((y[mask]==1).mean())})
                report['theia_test_labels_by_type']=type_table
                indeg=np.bincount(dst,minlength=len(nt));outdeg=np.bincount(src,minlength=len(nt))
                simple={}
                for title,score in [('total_degree_high',degree),('total_degree_low',-degree),('incoming_degree_high',indeg),('outgoing_degree_high',outdeg)]:
                    simple[title]=ranking(y,score)
                # Label-blind reference: rarity of exact feature signatures in training.
                test_payload=(rows.copy(),y.copy(),unique.copy(),counts.copy())
                report['theia_shortcut_rankings']=simple
        graph_stats.append(stats)
        print('Audited',name,stats['nodes'],'nodes',flush=True)
    report['theia_graphs']=graph_stats
    train_unique=np.unique(np.concatenate(train_unique))
    rows,y,test_unique,counts=test_payload
    seen=np.isin(rows,train_unique)
    report['theia_feature_overlap']={'unique_train_signatures':len(train_unique),'test_nodes_matching_training_features':int(seen.sum()),
        'fraction_test_nodes_matching_training_features':float(seen.mean()),
        'test_nodes_matching_validation_features':int(np.isin(rows,validation_unique).sum()),
        'note':'Feature equality is not proof of duplicated entities or train/test leakage.'}
    report['theia_shortcut_rankings']['unseen_training_signature']=ranking(y,(~seen).astype(int))
    report['theia_test_positive_prevalence']=float((y==1).mean())
    # Audit the exact StreamSpot prepared dataset used in the comparison.
    sp=ROOT/'data/processed/edges512_sample77_split2026';manifest=json.loads((sp/'manifest.json').read_text())
    graphs=json.loads((sp/'graphs.json').read_text());splits=json.loads((sp/'splits.json').read_text());stats=json.loads((sp/'statistics.json').read_text())
    sreport={'graph_count':len(graphs),'labels':distribution([g['y'] for g in graphs]),
        'prepared_hashes_match':{name:sha256(sp/name)==expected for name,expected in manifest['processed_sha256'].items()},
        'raw_archive_hash_matches_manifest':sha256(ROOT/'data/raw/all.tar.gz')==manifest['source_sha256'],
        'split_counts':{p:len(v) for p,v in splits.items()},
        'split_graph_ids_disjoint':all(not(set(splits[a])&set(splits[b])) for a,b in [('train','validation'),('train','test'),('validation','test')]),
        'raw_edges':sum(g['raw_edges'] for g in stats),'retained_edges':sum(g['sampled_edges'] for g in stats)}
    sreport['edge_retention_fraction']=sreport['retained_edges']/sreport['raw_edges']
    ratios=[g['sampled_edges']/g['raw_edges'] for g in stats]
    sreport['per_graph_edge_retention_quantiles']={str(q):float(np.quantile(ratios,q)) for q in [0,.5,1]}
    fingerprints={};ss_stats=[]
    with np.load(sp/'graphs.npz',allow_pickle=False) as archive:
        for g in graphs:
            gid=g['id'];x=archive[f'{gid}_x'];src=archive[f'{gid}_src'];dst=archive[f'{gid}_dst']
            if not np.isfinite(x).all() or (len(src) and (min(src.min(),dst.min())<0 or max(src.max(),dst.max())>=len(x))):raise ValueError('Malformed StreamSpot graph')
            h=hashlib.sha256()
            for arr in [x,src,dst]:h.update(str(arr.shape).encode());h.update(arr.tobytes())
            fingerprints.setdefault(h.hexdigest(),[]).append(gid)
            ss_stats.append({'graph_id':gid,'sampled_nodes':len(x),'sampled_edges':len(src),'isolated_nodes':int((np.bincount(np.concatenate([src,dst]),minlength=len(x))==0).sum())})
    sreport['all_features_finite_and_endpoints_valid']=True
    sreport['exact_serialized_duplicate_graph_groups']=[ids for ids in fingerprints.values() if len(ids)>1]
    sreport['duplicate_check_limit']='Exact tensor order only, not graph isomorphism or entity identity.'
    sreport['isolated_nodes_total']=sum(g['isolated_nodes'] for g in ss_stats)
    lookup={g['graph_id']:g for g in stats};ys={g['id']:g['y'] for g in graphs};test=splits['test']
    sreport['test_shortcut_rankings']={k:ranking([ys[i] for i in test],[lookup[i][k] for i in test]) for k in ['sampled_nodes','raw_edges']}
    sreport['shortcut_note']='Raw edge count is a source diagnostic, not necessarily a model input. Test-label diagnostic, not a newly validated detector.'
    report['streamspot']=sreport
    prediction_subgroups(run,report)
    report['audit_seconds']=time.perf_counter()-started
    save_json(out/'audit.json',report)
    print('Audit saved:',out/'audit.json',flush=True)


if __name__=='__main__':main()
