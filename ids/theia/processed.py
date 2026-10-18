"""Training on preprocessed THEIA DGL exports, without importing DGL.

This is a separate static-graph protocol: no invented timestamps or UUIDs.
Only allowlisted inert DGL records and weights-only CPU tensor storage are read.
"""
import argparse
import collections
import csv
import io
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import fingerprint, peak_rss_mib, save_json, sha256
from .experiment import MODELS, reevaluate, verify_checkpoints


class Record:
    def __new__(cls, *args):
        return super().__new__(cls)

    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state
        if isinstance(state, dict):
            self.__dict__.update(state)


def new_record(*args):
    return Record(*args)


def scheme_attribute(cls, name):
    if cls is not Record or name != '_reconstruct_scheme':
        raise pickle.UnpicklingError('Unexpected attribute lookup')
    return new_record


def load_storage(blob):
    return torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)


class GraphReader(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.tensors = []

    def tensor(self, storage, offset, size, stride, requires_grad, hooks):
        if requires_grad or hooks:
            raise pickle.UnpicklingError('Unexpected tensor hooks or gradients')
        value = torch._utils._rebuild_tensor_v2(storage, offset, size, stride, False, collections.OrderedDict())
        self.tensors.append(value)
        return value

    def find_class(self, module, name):
        records = {
            ('dgl.heterograph', 'DGLGraph'),
            ('dgl.heterograph_index', 'HeteroGraphIndex'),
            ('dgl.heterograph_index', 'HeteroPickleStates'),
            ('dgl.frame', 'Frame'), ('dgl.frame', 'Column'), ('dgl.frame', 'Scheme'),
        }
        if (module, name) in records:
            return Record
        allowed = {
            ('dgl._ffi.object', '_new_object'): new_record,
            ('builtins', 'bytearray'): bytearray,
            ('builtins', 'getattr'): scheme_attribute,
            ('collections', 'OrderedDict'): collections.OrderedDict,
            ('torch.storage', '_load_from_bytes'): load_storage,
            ('torch._utils', '_rebuild_tensor_v2'): self.tensor,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')
        return allowed[module, name]


def read_graph(path):
    with Path(path).open('rb') as stream:
        reader = GraphReader(stream)
        graph = reader.load()
        if stream.read(1):
            raise ValueError('Trailing bytes in graph pickle')
    if len(reader.tensors) != 6:
        raise ValueError('Expected exactly six tensors in the supported DGL export')
    src, dst, nt, nx, et, ex = [t.numpy() for t in reader.tensors]
    # Confirm tensor order against the actual node/edge frames, not just shapes.
    if len(graph._node_frames) != 1 or len(graph._edge_frames) != 1:
        raise ValueError('Only homogeneous graphs are supported')
    for frame, expected in [(graph._node_frames[0], (nt, nx)), (graph._edge_frames[0], (et, ex))]:
        if set(frame._columns) != {'type', 'attr'}:
            raise ValueError('Unexpected feature columns')
        for key, value in zip(('type', 'attr'), expected):
            if not np.array_equal(frame._columns[key].storage.numpy(), value):
                raise ValueError('Tensor/frame alignment mismatch')
    n = len(nt); e = len(src)
    if n == 0 or e == 0 or src.shape != (e,) or dst.shape != (e,) or nt.shape != (n,) or et.shape != (e,):
        raise ValueError('Invalid graph dimensions')
    if any(v.dtype != np.int64 for v in (src, dst, nt, et)):
        raise ValueError('Expected int64 endpoints and types')
    if min(src.min(), dst.min(), nt.min(), et.min()) < 0 or max(src.max(), dst.max()) >= n:
        raise ValueError('Invalid endpoint or type index')
    for typ, attr in [(nt, nx), (et, ex)]:
        if attr.ndim != 2 or len(attr) != len(typ) or typ.max() >= attr.shape[1]:
            raise ValueError('Invalid feature shape')
        if not np.isfinite(attr).all() or not np.isin(attr, [0, 1]).all() or not (attr.sum(1) == 1).all() or not np.array_equal(attr.argmax(1), typ):
            raise ValueError('Type attributes are not aligned one-hot vectors')
    return src.copy(), dst.copy(), nt.copy(), et.copy(), nx.shape[1], ex.shape[1]


def features(src, dst, edge_types, n, vocabulary):
    # Type itself is a prediction target, never an input feature.
    bins = np.array([vocabulary.get(int(t), 0) for t in edge_types], dtype=np.int64)
    width = len(vocabulary) + 1
    x = np.zeros((n, 2 * width), dtype=np.float32)
    np.add.at(x, (src, bins), 1)
    np.add.at(x, (dst, bins + width), 1)
    return np.log1p(x)


def prepare(source, output, config, smoke_nodes=None):
    source = Path(source); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    meta = json.loads((source/'metadata.json').read_text())
    if meta['n_train'] != 4 or meta['n_test'] != 1:
        raise ValueError('This protocol requires four training exports and one test export')
    positives = meta['malicious'][0]
    if not positives or any(type(i) is not int for i in positives) or len(set(positives)) != len(positives):
        raise ValueError('Invalid malicious node index list')
    if len(meta['malicious'][1]) != len(positives):
        raise ValueError('Malicious metadata columns are misaligned')
    names = ['train0.pkl', 'train1.pkl', 'train2.pkl', 'train3.pkl', 'test0.pkl']
    sources = {name: sha256(source/name) for name in names + ['metadata.json']}
    edge_vocab = {}; type_vocab = {}
    for name in names[:3]:
        src, dst, nt, et, nd, ed = read_graph(source/name)
        for v in sorted(set(et.tolist())):
            if int(v) not in edge_vocab: edge_vocab[int(v)] = len(edge_vocab)+1
        for v in sorted(set(nt.tolist())):
            if int(v) not in type_vocab: type_vocab[int(v)] = len(type_vocab)+1
    shards = []; partitions = {}; counts = []
    with (output/'targets.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['uuid', 'partition', 'window', 'cutoff_ns', 'label', 'type', 'label_source'])
        writer.writeheader()
        for window, name in enumerate(names):
            src, dst, nt, et, nd, ed = read_graph(source/name)
            if nd != meta['node_feature_dim'] or ed != meta['edge_feature_dim']:
                raise ValueError('Metadata feature dimensions disagree with graph')
            original_n, original_e = len(nt), len(src)
            if name == 'test0.pkl' and (min(positives) < 0 or max(positives) >= original_n):
                raise ValueError('Malicious node index outside test graph')
            if smoke_nodes:
                keep = (src < smoke_nodes) & (dst < smoke_nodes)
                src, dst, et = src[keep], dst[keep], et[keep]
                nt = nt[:smoke_nodes]
            part = 'train' if window < 3 else ('validation' if window == 3 else 'test')
            n = len(nt)
            x = features(src, dst, et, n, edge_vocab)
            ids = np.asarray([f'{name}:node:{i}' for i in range(n)])
            y = np.full(n, -1 if part == 'train' else 0, dtype=np.int64)
            if part == 'test':
                y[np.asarray([i for i in positives if i < n], dtype=np.int64)] = 1
            types = np.asarray([type_vocab.get(int(t), 0) for t in nt], dtype=np.int64)
            shard = f'{part}_{window}.npz'
            np.savez_compressed(output/shard, x=x, src=src, dst=dst, types=types, y=y,
                                uuids=ids, target=np.ones(n, dtype=bool))
            shards.append(dict(file=shard, partition=part, window=window, max_ns=None, nodes=n, edges=len(src), sha256=sha256(output/shard)))
            count = partitions.setdefault(part, {'targets': 0, 'edges': 0})
            count['targets'] += n; count['edges'] += len(src)
            for i in range(n):
                writer.writerow(dict(uuid=ids[i], partition=part, window=window, cutoff_ns='', label=int(y[i]), type=int(nt[i]),
                    label_source='unused_training_label' if part == 'train' else ('upstream_positive_index' if y[i] == 1 else 'presumed_background')))
            counts.append(dict(file=name, original_nodes=original_n, original_edges=original_e, used_nodes=n, used_edges=len(src)))
            print(f'Prepared {name}: {n:,} nodes, {len(src):,} edges -> {part}', flush=True)
    manifest = dict(format_version='processed_static_v1', dataset='theia_preprocessed', target='graph_local_node',
        sources=sources, config=config, shards=shards, partitions=partitions, counts=counts,
        type_vocabulary={str(k):v for k,v in type_vocab.items()},
        edge_vocabulary={str(k):v for k,v in edge_vocab.items()}, feature_dim=2*(len(edge_vocab)+1),
        targets_sha256=sha256(output/'targets.csv'), smoke_nodes=smoke_nodes,
        protocol='train0-2 fit; train3 validation presumed background; test0 upstream positives, complement presumed background. Graph-local IDs; cross-graph entity overlap unknown. No timestamps available.',
        provenance='User-supplied MAGIC-format graphs; upstream archive identity not independently verified.',
        prepare_seconds=time.perf_counter()-start, prepare_process_peak_rss_mib=peak_rss_mib())
    manifest['fingerprint'] = fingerprint(manifest)
    save_json(output/'manifest.json', manifest)
    save_json(output/'splits.json', {'train': names[:3], 'validation': names[3:4], 'test': names[4:], 'protocol': manifest['protocol']})
    validate_processed(output)
    return output


def validate_processed(data):
    data = Path(data); m = json.loads((data/'manifest.json').read_text())
    identity = dict(m); expected = identity.pop('fingerprint')
    if fingerprint(identity) != expected or sha256(data/'targets.csv') != m['targets_sha256']:
        raise ValueError('Prepared data fingerprint mismatch')
    totals = collections.Counter()
    for entry in m['shards']:
        if sha256(data/entry['file']) != entry['sha256']:
            raise ValueError('Prepared shard changed')
        with np.load(data/entry['file'], allow_pickle=False) as g:
            n = len(g['x'])
            if g['x'].shape != (n, m['feature_dim']) or not np.isfinite(g['x']).all():
                raise ValueError('Invalid prepared features')
            if any(len(g[k]) != n for k in ['types','target','y','uuids']):
                raise ValueError('Misaligned node arrays')
            if not np.isin(g['y'], [-1,0,1]).all() or len(g['src']) != len(g['dst']):
                raise ValueError('Invalid labels or edge arrays')
            if len(g['src']) and (min(g['src'].min(),g['dst'].min()) < 0 or max(g['src'].max(),g['dst'].max()) >= n):
                raise ValueError('Invalid prepared endpoint')
            totals[entry['partition']] += int(g['target'].sum())
    if dict(totals) != {p: v['targets'] for p,v in m['partitions'].items()}:
        raise ValueError('Partition counts disagree')
    return {'verified': True, 'targets': dict(totals), 'fingerprint': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/theia_processed.json')
    parser.add_argument('--source', default='data/theia')
    parser.add_argument('--smoke', action='store_true', help='One epoch on first 2048 nodes per actual graph; engineering check only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root/args.config).read_text())
    if args.smoke:
        config.update(epochs=1, rf_trees=8, rf_max_train_nodes=1024, max_reference_nodes=256, experiment_role='real_data_subset_engineering_check')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix = 'processed_smoke' if args.smoke else 'processed_full'
    out = root/'results'/'theia'/f'{prefix}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'config.json', config)
    print('Results:', out, flush=True)
    print('Protocol: train0-2 fit, train3 validation, test0 held-out. Non-positive evaluation nodes are presumed background.', flush=True)
    data = prepare(root/args.source, out/'data', config, 2048 if args.smoke else None)
    (out/'data_path.txt').write_text(str(data))
    for name in ['manifest.json', 'splits.json', 'targets.csv']:
        import shutil
        shutil.copyfile(data/name, out/name)
    save_json(out/'code_fingerprints.json', {str(p.relative_to(root)):sha256(p) for p in list((root/'ids'/'theia').glob('*.py'))+[root/'ids'/'models.py']})
    for seed in config['seeds']:
        for model in MODELS:
            print(f'Starting {model}, seed {seed}', flush=True)
            with (out/f'{model}_{seed}.log').open('w') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'ids.theia.worker', '--data', str(data), '--output', str(out), '--model', model, '--seed', str(seed)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f'Training failed; see {log.name}')
    reevaluate(out)
    if args.smoke:
        verify_checkpoints(out)
    print(f'Completed. Metrics: {out / "metrics.csv"}', flush=True)


if __name__ == '__main__':
    main()
