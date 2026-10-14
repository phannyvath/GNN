"""StreamSpot import. Labels are graph labels, never node labels or features."""
import hashlib, json, tarfile, urllib.request
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

URL = 'https://raw.githubusercontent.com/sbustreamspot/sbustreamspot-data/master/all.tar.gz'
SCENARIOS = ['YouTube', 'GMail', 'VGame', 'Drive-by-download attack', 'Download', 'CNN']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(root):
    path = Path(root) / 'data/raw/all.tar.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix('.part')
        try:
            urllib.request.urlretrieve(URL, tmp)
            with tarfile.open(tmp, 'r:gz') as tf:
                if not any(m.name.endswith('all.tsv') and m.isfile() for m in tf.getmembers()):
                    raise ValueError('Archive has no all.tsv')
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    print('Dataset:', path, flush=True)
    return path


def partitions(ids, split_seed):
    # Stratify by scenario, not graph-ID feature. Every whole graph belongs to one split.
    ids = np.asarray(sorted(ids))
    if set(ids.tolist()) != set(range(600)):
        raise ValueError('Expected all 600 StreamSpot graphs. Partial datasets are not accepted.')
    train, rest = train_test_split(ids, test_size=.4, random_state=split_seed, stratify=ids//100)
    val, test = train_test_split(rest, test_size=.5, random_state=split_seed, stratify=rest//100)
    return {k: sorted(v.tolist()) for k,v in [('train',train),('validation',val),('test',test)]}


def prepare(root, config):
    root = Path(root)
    source = root / 'data/raw/all.tar.gz'
    if not source.exists():
        raise FileNotFoundError('Run: python main.py download')
    key = f"edges{config['max_edges']}_sample{config['reservoir_seed']}_split{config['split_seed']}"
    target = root / 'data/processed' / key
    if (target/'manifest.json').exists():
        if json.loads((target/'manifest.json').read_text())['source_sha256'] != sha256(source):
            raise ValueError('Raw data changed; remove this processed cache and prepare again.')
        return target
    target.mkdir(parents=True, exist_ok=True)
    cap = int(config['max_edges'])
    # Keep independent uniform priorities per graph. Prune in chunks, preserving global row order.
    rng = [np.random.default_rng(config['reservoir_seed']+i) for i in range(600)]
    rows = [[] for _ in range(600)]
    counts = np.zeros(600,dtype=np.int64)
    total = 0
    def prune(g):
        a = rows[g]
        if len(a) > cap:
            priorities = np.fromiter((r[0] for r in a),dtype=float)
            keep = np.argpartition(priorities, cap-1)[:cap]
            rows[g] = [a[i] for i in keep]
    with tarfile.open(source, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.name.endswith('all.tsv') and m.isfile()]
        if len(members)!=1:
            raise ValueError('Expected one all.tsv archive member')
        with archive.extractfile(members[0]) as f:
            for line in f:
                fields = line.decode('utf-8').rstrip('\r\n').split('\t')
                if len(fields)!=6:
                    raise ValueError(f'Malformed row {total+1}: expected six columns')
                s,st,d,dt,et,gtext = fields
                g = int(gtext)
                if not 0<=g<600: raise ValueError('Invalid graph ID')
                ordinal = int(counts[g]); counts[g]+=1; total+=1
                rows[g].append((float(rng[g].random()),ordinal,s,st,d,dt,et))
                if len(rows[g]) >= cap+8192: prune(g)
                if total % 5000000 == 0: print(f'Parsed {total:,} edges',flush=True)
    for g in range(600): prune(g); rows[g].sort(key=lambda r:r[1])
    splits = partitions([g for g in range(600) if counts[g]>0],config['split_seed'])
    # Vocabulary fitted ONLY on training graphs; index 0 is unseen/unknown.
    ntypes=sorted({r[j] for g in splits['train'] for r in rows[g] for j in [3,5]})
    etypes=sorted({r[6] for g in splits['train'] for r in rows[g]})
    nv={t:i+1 for i,t in enumerate(ntypes)}; ev={t:i+1 for i,t in enumerate(etypes)}
    graphs=[]; stats=[]
    for g in range(600):
        mapping={}; types=[]; src=[]; dst=[]; rel=[]; order=[]
        for _,ordinal,s,st,d,dt,et in rows[g]:
            for node,typ in [(s,st),(d,dt)]:
                if node not in mapping: mapping[node]=len(types); types.append(nv.get(typ,0))
                elif types[mapping[node]] != nv.get(typ,0):
                    raise ValueError(f'Conflicting types for node {node} in graph {g}')
            src.append(mapping[s]); dst.append(mapping[d]); rel.append(ev.get(et,0)); order.append(ordinal)
        n=len(types); src=np.asarray(src); dst=np.asarray(dst); rel=np.asarray(rel)
        # No ID, label, scenario or ordering information enters x.
        x=np.zeros((n,len(nv)+1+2*(len(ev)+1)),dtype=np.float32)
        x[np.arange(n),types]=1
        offset=len(nv)+1; width=len(ev)+1
        np.add.at(x,(src,offset+rel),1)
        np.add.at(x,(dst,offset+width+rel),1)
        x[:,offset:]=np.log1p(x[:,offset:])
        graph={'id':g,'y':int(300<=g<400),'x':x,'src':src.astype(np.int64),
               'dst':dst.astype(np.int64),'node_ids':list(mapping),'types':types,
               'edge_types':rel.tolist(),'edge_ordinals':order,'raw_edges':int(counts[g])}
        graphs.append(graph)
        stats.append({'graph_id':g,'scenario':SCENARIOS[g//100],'label':graph['y'],
                      'raw_edges':int(counts[g]),'sampled_edges':len(src),'sampled_nodes':n})
    # Compressed array storage, no executable pickle input.
    arrays={}
    for graph in graphs:
        g=graph['id']
        for k in ['x','src','dst']: arrays[f'{g}_{k}']=graph[k]
    np.savez_compressed(target/'graphs.npz',**arrays)
    meta=[{k:v for k,v in graph.items() if k not in ['x','src','dst']} for graph in graphs]
    (target/'graphs.json').write_text(json.dumps(meta))
    (target/'splits.json').write_text(json.dumps(splits,indent=2))
    (target/'statistics.json').write_text(json.dumps(stats,indent=2))
    manifest={'dataset':'StreamSpot ALL','source_url':URL,'source_sha256':sha256(source),
        'raw_edge_count':total,'graph_count':600,'feature_dim':graphs[0]['x'].shape[1],
        'node_type_vocabulary':nv,'edge_type_vocabulary':ev,'config':{k:config[k] for k in ['max_edges','reservoir_seed','split_seed']},
        'processed_sha256':{name:sha256(target/name) for name in ['graphs.npz','graphs.json','splits.json']},
        'sampling':'Uniform random edge reservoir per graph; original row ordinals retained only as evidence',
        'detection_unit':'whole graph','timestamps_available':False,'names_available':False,
        'split':'60/20/20 whole graphs, stratified by scenario; not chronological',
        'limitations':['No node-level ground truth','No real timestamps or semantic file names',
                      'Edge sampling removes paths and changes degree distributions',
                      'Scenario stratification does not test unseen attack families']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Prepared {total:,} raw edges into {target}',flush=True)
    return target


def load(target):
    target=Path(target)
    manifest=json.loads((target/'manifest.json').read_text())
    for name,fingerprint in manifest.get('processed_sha256',{}).items():
        if sha256(target/name)!=fingerprint: raise ValueError(f'Processed file changed: {name}')
    meta=json.loads((target/'graphs.json').read_text())
    with np.load(target/'graphs.npz',allow_pickle=False) as arrays:
        graphs=[dict(m,**{k:arrays[f"{m['id']}_{k}"] for k in ['x','src','dst']}) for m in meta]
    return graphs,json.loads((target/'splits.json').read_text()),json.loads((target/'manifest.json').read_text())
