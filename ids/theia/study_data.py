"""Data contracts for the reproducible static-export study."""
import csv
import json
from pathlib import Path
import numpy as np
from .common import fingerprint, save_json, sha256
from .processed import prepare, read_graph, validate_processed

FILES = {'train':['train0.pkl','train1.pkl','train2.pkl'], 'validation':['train3.pkl'], 'test':['test0.pkl']}


def validate_config(c):
    for part,files in FILES.items():
        if c[f'{part}_files'] != files: raise ValueError('This protocol requires fixed graph-file splits')
    if c['device'] != 'cpu' or c['benign_only_training'] or c['training_node_sampling']!='none':
        raise ValueError('Only CPU, all eligible training nodes, no retrospective training-label filtering supported')
    if len(set(c['seeds'])) != len(c['seeds']) or len(c['seeds'])<3: raise ValueError('Use at least three distinct seeds')
    for k in ['epochs','hidden','threads','patience','rf_trees','rf_max_depth','knn_k','query_batch_nodes','max_nodes_per_graph_batch','max_edges_per_graph_batch']:
        if type(c[k]) is not int or c[k]<=0: raise ValueError(f'Invalid {k}')
    if not 0<c['mask_rate']<=1 or not 0<c['validation_fpr']<1 or c['min_delta']<0 or c['learning_rate']<=0: raise ValueError('Invalid learning/selection settings')
    if c['smoke_nodes'] is not None and (type(c['smoke_nodes']) is not int or c['smoke_nodes']<2): raise ValueError('Invalid smoke cap')


def make_data(source, data, c):
    validate_config(c)
    prepare(source,data,c,c['smoke_nodes'])
    m=json.loads((data/'manifest.json').read_text())
    audit_path=Path(__file__).resolve().parents[2]/'results/data_quality/audit.json'
    evidence=None
    if audit_path.exists():
        candidate=json.loads(audit_path.read_text()).get('theia_provenance',{})
        if candidate.get('exact_upstream_match') and candidate.get('local_archive_sha256')==sha256(source/'graphs.zip'):
            evidence=candidate
    m['provenance']=evidence or {'status':'Not verified against upstream in this run'}
    m['class_counts']={}
    for entry,g in batches(data,None,c):
        m['class_counts'][entry['partition']]=m['class_counts'].get(entry['partition'],{'positive':0,'presumed_background':0,'unknown':0})
        for key,label in [('positive',1),('presumed_background',0),('unknown',-1)]:m['class_counts'][entry['partition']][key]+=int((g['y']==label).sum())
    m['node_identity']='Graph-local node occurrences; unique entities across graphs and overlap are unknown.'
    m.pop('fingerprint');m['fingerprint']=fingerprint(m);save_json(data/'manifest.json',m)
    result=verify_data(data,source,c)
    save_json(data/'alignment_audit.json',result)
    return m


def batches(data, partition, config):
    data=Path(data);m=json.loads((data/'manifest.json').read_text())
    for entry in m['shards']:
        if partition is not None and entry['partition']!=partition: continue
        if entry['nodes']>config['max_nodes_per_graph_batch'] or entry['edges']>config['max_edges_per_graph_batch']:
            raise ValueError('Graph exceeds declared memory budget; no silent sampling. Use a reviewed batching protocol.')
        with np.load(data/entry['file'],allow_pickle=False) as archive:
            g={k:archive[k] for k in archive.files}
        yield entry,g


def verify_data(data,source,c):
    validate_config(c);data=Path(data);source=Path(source)
    validate_processed(data)
    m=json.loads((data/'manifest.json').read_text())
    if m['config']!=c:raise ValueError('Prepared config differs from run config')
    for name,digest in m['sources'].items():
        if sha256(source/name)!=digest:raise ValueError(f'Source changed: {name}')
    meta=json.loads((source/'metadata.json').read_text());positive=np.asarray(meta['malicious'][0],dtype=np.int64)
    if len(positive)!=len(np.unique(positive)) or (positive<0).any():raise ValueError('Invalid positive indices')
    seen=set();partitions={};vocab={int(k):v for k,v in m['type_vocabulary'].items()}
    with (data/'targets.csv').open() as stream:
        reader=csv.DictReader(stream)
        for entry,g in batches(data,None,c):
            part=entry['partition'];n=len(g['x']);original=m['counts'][entry['window']]['file']
            if original not in FILES[part]:raise ValueError('Graph assigned to wrong split')
            src,dst,types,edge_types,_,_=read_graph(source/original)
            retained=len(types) if c['smoke_nodes'] is None else min(len(types),c['smoke_nodes'])
            if n!=retained:raise ValueError('Unexpected target subsampling')
            if original=='test0.pkl' and (positive.max()>=len(types)):raise ValueError('Positive index out of range')
            expected_y=np.full(n,-1 if part=='train' else 0,dtype=np.int64)
            if part=='test':expected_y[positive[positive<n]]=1
            if not np.array_equal(g['y'],expected_y):raise ValueError('Malicious-index/label alignment failure')
            if not np.array_equal(g['types'],[vocab.get(int(t),0) for t in types[:n]]):raise ValueError('Node type misalignment')
            keep=(src<n)&(dst<n)
            if not np.array_equal(g['src'],src[keep]) or not np.array_equal(g['dst'],dst[keep]):raise ValueError('Edge alignment failure')
            if not g['target'].all():raise ValueError('Unexpected target exclusions')
            for i,uid in enumerate(g['uuids']):
                if uid!=f'{original}:node:{i}' or uid in seen:raise ValueError('Node ID alignment or split overlap')
                seen.add(str(uid));row=next(reader,None)
                if row is None or row['uuid']!=uid or row['partition']!=part or int(row['label'])!=g['y'][i]:raise ValueError('Target CSV alignment failure')
            partitions[part]=partitions.get(part,0)+n
        if next(reader,None) is not None:raise ValueError('Extra target rows')
    return {'verified':True,'graph_local_target_ids_disjoint':True,'unique_entity_disjointness':'unverifiable without UUID mappings',
            'labels_match_upstream_indices':True,'all_expected_nodes_and_edges_retained':c['smoke_nodes'] is None,
            'partition_node_occurrences':partitions,'fingerprint':m['fingerprint']}
