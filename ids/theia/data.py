"""CDM18 -> disk SQLite -> bounded chronological graph shards.

Entity types are immutable schema metadata, never later paths/names. Dynamic
features use only events within a shard and alerts are issued at its end.
"""
import csv
import gzip
import io
import json
import re
import sqlite3
import tarfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .common import fingerprint, peak_rss_mib, save_json, sha256

SCHEMA = 'com.bbn.tc.schema.avro.cdm18.'
ENTITY_KINDS = {'Subject', 'FileObject', 'NetFlowObject', 'MemoryObject', 'UnnamedPipeObject', 'Principal', 'SrcSinkObject'}
TARGET_KINDS = {'SUBJECT_PROCESS', 'FILE_OBJECT_BLOCK', 'FILE_OBJECT_CHAR', 'FILE_OBJECT_DIR', 'FILE_OBJECT_FILE', 'FileObject', 'NetFlowObject'}
PARTS = ('train', 'validation', 'test')


def unwrap(value):
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return value


def uid(value):
    value = unwrap(value)
    if value is None or value == 'null': return None
    try: result = str(uuid.UUID(str(value))).upper()
    except (ValueError, AttributeError) as e: raise ValueError(f'Invalid CDM UUID: {value!r}') from e
    return None if result == '00000000-0000-0000-0000-000000000000' else result


@contextmanager
def records(path, max_line_bytes):
    """Yield (member, line number, raw bytes); no filesystem extraction."""
    path = Path(path)
    def lines(stream, name):
        number = 0
        while True:
            line = stream.readline(max_line_bytes + 1)
            if not line: break
            number += 1
            if len(line) > max_line_bytes: raise ValueError(f'Oversized line: {name}:{number}')
            if line.strip(): yield name, number, line
    if path.name.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(path, 'r|gz') as tar:
            def members():
                for member in tar:
                    if member.isfile() and re.search(r'\.json(?:\.\d+)?$', member.name):
                        with tar.extractfile(member) as stream: yield from lines(stream, member.name)
            yield members()
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream: yield lines(stream, path.name)


def import_raw(paths, database, config):
    db = sqlite3.connect(database)
    db.executescript('''PRAGMA cache_size=-32768; PRAGMA temp_store=FILE;
    CREATE TABLE entities(uuid TEXT PRIMARY KEY, type TEXT NOT NULL);
    CREATE TABLE events(event_id TEXT, slot INTEGER, ts INTEGER, src TEXT, dst TEXT, type TEXT,
                        PRIMARY KEY(event_id, slot));
    CREATE TABLE assigned(uuid TEXT PRIMARY KEY, partition TEXT, window INTEGER);
    ''')
    try:
        counts = Counter(); sources = []
        for path in paths:
            path = Path(path); source = {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}
            provenance = Path(str(path) + '.source.json')
            if provenance.exists(): source['download'] = json.loads(provenance.read_text())
            sources.append(source)
            with records(path, config['max_line_bytes']) as rows:
                for member, lineno, line in rows:
                    if counts['records'] >= config['max_raw_records']:
                        counts['record_limit_reached'] = 1; break
                    counts['records'] += 1
                    try:
                        row = json.loads(line); datum = row['datum']
                        if len(datum) != 1: raise ValueError('Expected one Avro datum union branch')
                        key, value = next(iter(datum.items()))
                        if not key.startswith(SCHEMA): raise ValueError(f'Unsupported schema: {key}')
                        kind = key[len(SCHEMA):]; counts[kind] += 1
                        if kind in ENTITY_KINDS:
                            node = uid(value.get('uuid'))
                            if node is None: counts['null_entity'] += 1; continue
                            typ = str(unwrap(value.get('type', kind)))
                            old = db.execute('SELECT type FROM entities WHERE uuid=?', (node,)).fetchone()
                            if old and old[0] != typ: raise ValueError(f'Conflicting immutable type for {node}')
                            if old is None:
                                if counts['entities_imported'] >= config['max_import_nodes']:
                                    counts['entities_over_limit'] += 1; continue
                                db.execute('INSERT INTO entities VALUES (?,?)', (node, typ))
                                counts['entities_imported'] += 1
                        elif kind == 'Event':
                            src = uid(value.get('subject')); event_id = uid(value.get('uuid'))
                            if not event_id: raise ValueError('Event missing non-null UUID')
                            ts = unwrap(value['timestampNanos'])
                            if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0: raise ValueError('Invalid integer nanoseconds')
                            typ = str(unwrap(value['type']))
                            for slot, name in enumerate(('predicateObject', 'predicateObject2')):
                                dst = uid(value.get(name))
                                if src is None or dst is None: counts['null_endpoint'] += 1; continue
                                if counts['edges_imported'] >= config['max_import_edges']:
                                    counts['edges_over_limit'] += 1; continue
                                a, b = (dst, src) if any(t in typ for t in ('READ', 'RECV', 'LOAD')) else (src, dst)
                                existing = db.execute('SELECT ts,src,dst,type FROM events WHERE event_id=? AND slot=?', (event_id, slot)).fetchone()
                                payload = (ts, a, b, typ)
                                if existing:
                                    if existing != payload: raise ValueError('Conflicting duplicate event UUID/slot')
                                    counts['duplicate_edges'] += 1; continue
                                db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)', (event_id, slot, *payload))
                                counts['edges_imported'] += 1
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f'{path}:{member}:{lineno}: {error}') from error
                    if counts['records'] % 10000 == 0: db.commit()
            db.commit()
        db.executescript('CREATE INDEX event_time ON events(ts,event_id,slot); CREATE INDEX event_src ON events(src); CREATE INDEX event_dst ON events(dst);')
        counts['entities'] = db.execute('SELECT COUNT(*) FROM entities').fetchone()[0]
        counts['missing_endpoint_edges'] = db.execute('''SELECT COUNT(*) FROM events e LEFT JOIN entities a ON a.uuid=e.src
             LEFT JOIN entities b ON b.uuid=e.dst WHERE a.uuid IS NULL OR b.uuid IS NULL''').fetchone()[0]
        return db, dict(counts), sources
    except BaseException:
        db.close()
        raise


def read_labels(malicious_path, curated_path=None):
    labels = {}; provenance = []
    if malicious_path:
        with open(malicious_path) as stream:
            for line in stream:
                if line.strip(): labels[uid(line.strip())] = (1, 'ThreaTrace entity-level positive; onset unspecified')
        provenance.append({'path': str(malicious_path), 'sha256': sha256(malicious_path)})
    if curated_path:
        with open(curated_path, newline='') as stream:
            for row in csv.DictReader(stream):
                node = uid(row['uuid']); y = int(row['label']); evidence = row['evidence'].strip()
                if not node or y not in (0, 1) or not evidence: raise ValueError('Curated labels need UUID, 0/1, evidence')
                if node in labels and labels[node][0] != y: raise ValueError(f'Conflicting label for {node}')
                labels[node] = (y, evidence)
        provenance.append({'path': str(curated_path), 'sha256': sha256(curated_path)})
    return labels, provenance


def boundaries(db, config):
    lo, hi = db.execute('SELECT MIN(ts),MAX(ts) FROM events').fetchone()
    if lo is None or lo == hi: raise ValueError('Need events at multiple timestamps')
    if config['split_mode'] == 'explicit':
        start, train_end, val_end, end = [int(config[k]) for k in ('start_ns','train_end_ns','validation_end_ns','end_ns')]
    elif config['split_mode'] == 'observed_time_fraction':
        start, end = lo, hi + 1
        train_end = start + (end-start)*60//100; val_end = start + (end-start)*80//100
    else: raise ValueError('Unknown split_mode')
    if not start < train_end < val_end < end: raise ValueError('Split boundaries must be strictly chronological')
    return [start, train_end, val_end, end]


def prepare(paths, output, config, malicious_path=None, curated_path=None):
    policy = config['label_policy']
    if policy not in ('positive_unknown', 'scoped_background'): raise ValueError('Unknown label policy')
    if policy == 'scoped_background' and not config.get('background_rationale'):
        raise ValueError('scoped_background requires an explicit background_rationale')
    started = time.perf_counter(); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', config)
    labels, label_sources = read_labels(malicious_path, curated_path)
    db, counts, sources = import_raw(paths, output / 'events.sqlite', config)
    target_file = None
    try:
        cuts = boundaries(db, config)
        # Vocabulary learned solely from events in the training interval, with UNK bin 0.
        edge_names = [r[0] for r in db.execute('SELECT DISTINCT type FROM events WHERE ts>=? AND ts<? ORDER BY type', cuts[:2])]
        type_names = [r[0] for r in db.execute('''SELECT DISTINCT n.type FROM entities n JOIN
            (SELECT src AS uuid FROM events WHERE ts>=? AND ts<? UNION SELECT dst FROM events WHERE ts>=? AND ts<?) e
            ON n.uuid=e.uuid ORDER BY n.type''', cuts[:2]*2)]
        edge_vocab = {name: i+1 for i,name in enumerate(edge_names)}
        type_vocab = {name: i+1 for i,name in enumerate(type_names)}
        shards = []; summary = {}; target_file = open(output/'targets.csv', 'w', newline='')
        fields = ['uuid','partition','window','cutoff_ns','label','label_source','type']
        writer = csv.DictWriter(target_file, fieldnames=fields); writer.writeheader()
        dim = 2*(len(edge_vocab)+1) + 2
        for part, start, end in zip(PARTS, cuts, cuts[1:]):
            stats = Counter(); edges = []; nodes = {}; bucket = None; window = 0
            def flush():
                nonlocal edges, nodes, window
                if not edges: return
                node_ids = list(nodes); x = np.zeros((len(nodes), dim), np.float32)
                src = np.asarray([nodes[e[0]][0] for e in edges], dtype=np.int64)
                dst = np.asarray([nodes[e[1]][0] for e in edges], dtype=np.int64)
                et = np.asarray([edge_vocab.get(e[2],0) for e in edges], dtype=np.int64)
                timestamps = np.asarray([e[3] for e in edges], dtype=np.int64)
                bins = len(edge_vocab)+1
                np.add.at(x, (src, et), 1); np.add.at(x, (dst, et+bins), 1)
                # Fractional within-window temporal position, no absolute date/UUID/label features.
                earliest = np.full(len(nodes), timestamps.max(), dtype=np.int64)
                latest = np.full(len(nodes), timestamps.min(), dtype=np.int64)
                for indexes in (src,dst):
                    np.minimum.at(earliest,indexes,timestamps); np.maximum.at(latest,indexes,timestamps)
                x[:,:-2] = np.log1p(x[:,:-2])
                duration = max(int(timestamps.max()-timestamps.min()),1)
                x[:,-2] = (latest-earliest)/duration
                x[:,-1] = (timestamps.max()-latest)/duration
                target = np.zeros(len(nodes), dtype=bool); y = np.full(len(nodes), -1, dtype=np.int8)
                kinds = []; label_reasons = []
                for index, node in enumerate(node_ids):
                    kind = nodes[node][1]; kinds.append(kind)
                    label, reason = labels.get(node, (-1, 'unknown: no explicit node label'))
                    if kind not in TARGET_KINDS:
                        label, reason = -1, 'ineligible entity type (context only)'
                    elif label == -1 and policy == 'scoped_background':
                        label, reason = 0, 'presumed background within declared imported scope; NOT verified benign'
                    # A UUID is a scored target only once, in its earliest retained window.
                    old = db.execute('SELECT partition FROM assigned WHERE uuid=?', (node,)).fetchone()
                    if kind in TARGET_KINDS and old is None:
                        db.execute('INSERT INTO assigned VALUES (?,?,?)', (node, part, window)); target[index] = True
                        stats['targets'] += 1; stats[f'label_{label}'] += 1
                        writer.writerow(dict(zip(fields, [node,part,window,int(timestamps.max()),label,reason,kind])))
                    y[index] = label; label_reasons.append(reason)
                filename = f'{part}_{window:05d}.npz'
                np.savez_compressed(output/filename, x=x, src=src, dst=dst, edge_types=et, timestamps=timestamps,
                    uuids=np.asarray(node_ids), types=np.asarray([type_vocab.get(k,0) for k in kinds],dtype=np.int64),
                    kind=np.asarray(kinds), target=target, y=y)
                shards.append({'file': filename, 'sha256': sha256(output/filename), 'partition': part, 'window': window,
                               'nodes':len(nodes), 'edges':len(edges), 'targets':int(target.sum()),
                               'min_ns':int(timestamps.min()),'max_ns':int(timestamps.max())})
                stats['windows'] += 1; stats['node_occurrences'] += len(nodes); stats['edges'] += len(edges)
                edges = []; nodes = {}; window += 1; db.commit()
            query = '''SELECT e.src,e.dst,e.type,e.ts,a.type,b.type FROM events e
                JOIN entities a ON a.uuid=e.src JOIN entities b ON b.uuid=e.dst
                WHERE e.ts>=? AND e.ts<? ORDER BY e.ts,e.event_id,e.slot'''
            for a,b,typ,ts,ak,bk in db.execute(query, (start,end)):
                new_bucket = (ts-start)//config['window_ns']
                additions = len({a,b} - nodes.keys())
                if edges and (new_bucket != bucket or len(edges) >= config['max_edges_per_window'] or
                              len(nodes)+additions > config['max_nodes_per_window']):
                    flush()
                if window >= config['max_windows_per_partition']:
                    stats['window_limit_reached'] = 1; break
                bucket = new_bucket
                for n,k in ((a,ak),(b,bk)):
                    if n not in nodes: nodes[n] = (len(nodes),k)
                edges.append((a,b,typ,ts))
            flush(); summary[part] = dict(stats)
        matched = sum(db.execute('SELECT 1 FROM assigned WHERE uuid=?',(node,)).fetchone() is not None for node in labels)
    finally:
        if target_file is not None: target_file.close()
        db.close()
    manifest = {'dataset':config.get('dataset','darpa_tc_e3_theia'),'target':'entity_once_at_first_retained_window_end',
        'format_version':1, 'sources':sources,'label_sources':label_sources,'counts':counts,
        'partitions':summary,'boundaries_ns':cuts,'config':config,'shards':shards,
        'edge_vocabulary':edge_vocab,'type_vocabulary':type_vocab,'feature_dim':dim,
        'explicit_label_count':len(labels),'explicit_labels_matched_to_targets':matched,
        'explicit_labels_unmatched_to_targets':len(labels)-matched,
        'targets_sha256':sha256(output/'targets.csv'),
        'training_label_use':'none: all training-window behaviour, potentially contaminated',
        'prepare_seconds':time.perf_counter()-started,'prepare_process_peak_rss_mib':peak_rss_mib()}
    identity = {k:manifest[k] for k in ('dataset','sources','label_sources','config','shards','targets_sha256','edge_vocabulary','type_vocabulary','feature_dim','boundaries_ns')}
    manifest['fingerprint'] = fingerprint(identity)
    save_json(output/'manifest.json',manifest)
    save_json(output/'splits.json',{'boundaries_ns':cuts,'intervals':'[start,end)',
        'target_assignment':'first retained occurrence; UUID targets disjoint; earlier entities may recur as context',
        'target_file':'targets.csv','sha256':manifest['targets_sha256']})
    validate(output)
    print(json.dumps({'prepared':str(output), 'counts':counts, 'partitions':summary},indent=2))
    return output


def validate(directory):
    directory = Path(directory); m = json.loads((directory/'manifest.json').read_text())
    identity = {k:m[k] for k in ('dataset','sources','label_sources','config','shards','targets_sha256','edge_vocabulary','type_vocabulary','feature_dim','boundaries_ns')}
    if fingerprint(identity) != m['fingerprint']: raise ValueError('Manifest fingerprint mismatch')
    seen = set(); expected = {}; cuts = m['boundaries_ns']; tally = Counter()
    with open(directory/'targets.csv',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['uuid'] in expected: raise ValueError('UUID appears in multiple target rows')
            expected[row['uuid']] = row
    if sha256(directory/'targets.csv') != m['targets_sha256']: raise ValueError('Targets fingerprint mismatch')
    for entry in m['shards']:
        if sha256(directory/entry['file']) != entry['sha256']: raise ValueError('Shard fingerprint mismatch')
        with np.load(directory/entry['file'],allow_pickle=False) as g:
            p = PARTS.index(entry['partition']); n = len(g['x'])
            if n > m['config']['max_nodes_per_window'] or len(g['src']) > m['config']['max_edges_per_window']:
                raise ValueError('Window exceeds budget')
            if not np.isfinite(g['x']).all(): raise ValueError('Nonfinite features')
            if len(set(g['uuids'])) != n: raise ValueError('Duplicate node IDs within window')
            if not np.isin(g['y'],[-1,0,1]).all(): raise ValueError('Invalid node label')
            if len(g['target']) != n or len(g['types']) != n: raise ValueError('Node arrays misaligned')
            if (g['timestamps'] < cuts[p]).any() or (g['timestamps'] >= cuts[p+1]).any(): raise ValueError('Split leakage')
            if len(g['src']) and max(g['src'].max(),g['dst'].max()) >= n: raise ValueError('Endpoint misalignment')
            for i in np.flatnonzero(g['target']):
                u = str(g['uuids'][i]); row = expected[u]
                if u in seen: raise ValueError('Target UUID overlap')
                seen.add(u)
                if row['partition'] != entry['partition'] or int(row['window']) != entry['window'] or int(row['label']) != g['y'][i] or int(row['cutoff_ns']) != entry['max_ns'] or row['type'] != g['kind'][i]:
                    raise ValueError('Label or split alignment failure')
                tally[entry['partition']] += 1
    if seen != expected.keys(): raise ValueError('Missing targets')
    return {'fingerprint':m['fingerprint'],'targets':dict(tally),'verified':True}


def windows(directory, part):
    directory = Path(directory); manifest = json.loads((directory/'manifest.json').read_text())
    for entry in manifest['shards']:
        if entry['partition'] == part:
            with np.load(directory/entry['file'],allow_pickle=False) as archive:
                yield entry, {key:archive[key] for key in archive.files}
