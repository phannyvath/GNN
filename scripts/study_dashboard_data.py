"""Evidence normalization for the offline dashboard. No model deserialization."""
import hashlib
import json
import math
from pathlib import Path


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def enrich(folder, rows):
    manifest = load(folder/'data/manifest.json', load(folder/'manifest.json', {}))
    splits = load(folder/'data/splits.json', load(folder/'splits.json', {}))
    config = load(folder/'config.json', {})
    counts = manifest.get('counts', [])
    if not isinstance(counts, list): counts = []
    partitions = []
    classes = manifest.get('class_counts', {})
    shards = manifest.get('shards', [])
    statistics = load(folder/'statistics.json', [])
    for split in ('train', 'validation', 'test'):
        part = manifest.get('partitions', {}).get(split, {})
        records = [r for r in statistics if r.get('graph_id') in splits.get(split, [])]
        cc = classes.get(split, {})
        partitions.append(dict(split=split, targets=part.get('targets',len(records) if records else None),
            graphs=sum(s.get('partition')==split for s in shards) if shards else len(records) if records else None,
            edges=part.get('edges',sum(r['sampled_edges'] for r in records) if records else None),
            positive=cc.get('positive',sum(r['label']==1 for r in records) if records else None),
            background=cc.get('presumed_background',sum(r['label']==0 for r in records) if records else None),
            unknown=cc.get('unknown',0 if records else None)))
    audit=load(folder/'evaluation_audit.json', {})
    complete=(folder/'study_complete.json').exists()
    state=load(folder/'run_status.json', {})
    valid_states={'queued','running','completed','failed','interrupted'}
    status=state.get('state') if state.get('state') in valid_states else 'completed' if complete else 'recorded results; completion unverified' if rows and not folder.name.startswith('study_') else 'incomplete; state unrecorded'
    compatible=bool(complete and audit.get('identical_targets_and_labels_all_methods_and_seeds') and audit.get('thresholds_validation_only') and audit.get('all_saved_metrics_recomputed'))
    fps={r.get('fingerprint') for r in rows}
    if len(fps)!=1 or None in fps or (manifest.get('fingerprint') and fps!={manifest['fingerprint']}): compatible=False
    seeds={r.get('seed') for r in rows}
    models={r.get('model') for r in rows}
    if len(rows)!=len(seeds)*len(models) or len({(r.get('model'),r.get('seed')) for r in rows})!=len(rows): compatible=False
    histories=[]
    environments={}
    for r in rows:
        tp,fp,fn=r.get('tp'),r.get('fp'),r.get('fn')
        if all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in (tp,fp,fn)):
            denominator=5*tp+4*fn+fp
            r['f2']=5*tp/denominator if denominator else None
        seconds=r.get('inference_seconds')
        targets=r.get('test_targets',r.get('known_targets',r.get('test_graphs')))
        r['throughput']=targets/seconds if targets is not None and seconds and seconds>0 else None
        worker=folder/f'{r["model"]}_{r["seed"]}'
        hist=load(worker/'epochs.json', [])
        if hist: histories.append(dict(model=r['model'],seed=r['seed'],epochs=hist))
        env=load(worker/'environment.json', {})
        if not env: env={k:r[k] for k in ('platform','device','python_version','torch_version','memory_definition') if k in r}
        environments[f'{r["model"]}/{r["seed"]}']=env
    return dict(schema_version=1,manifest=manifest,splits=splits,partitions=partitions,
        dataset_counts=dict(raw_records=manifest.get('raw_edge_count'),raw_record_unit='edges' if 'raw_edge_count' in manifest else None,
            original_node_occurrences=sum(c['original_nodes'] for c in counts) if counts else None,
            original_edges=sum(c['original_edges'] for c in counts) if counts else manifest.get('raw_edge_count'),
            graphs=len(counts) if counts else manifest.get('graph_count'),
            processed_node_occurrences=sum(c['used_nodes'] for c in counts) if counts else sum(r['sampled_nodes'] for r in statistics) if statistics else None,
            processed_edges=sum(c['used_edges'] for c in counts) if counts else sum(r['sampled_edges'] for r in statistics) if statistics else None,
            feature_dim=manifest.get('feature_dim'),unique_entities=None),
        subset=any(c['used_nodes']!=c['original_nodes'] or c['used_edges']!=c['original_edges'] for c in counts) or bool(statistics and sum(r['sampled_edges'] for r in statistics)<manifest.get('raw_edge_count',0)),
        run_state=status,state_evidence=state,compatible=compatible,
        compatibility_reason='Saved audit verifies shared targets/labels and validation-only thresholds; fingerprints and model/seed coverage agree.' if compatible else 'Insufficient evidence to recommend a winner: completion, shared targets/labels, fingerprints or complete model/seed coverage are not verified together.',
        histories=histories,environments=environments,run_date=load(folder/'invocation.json',{}).get('utc'),
        code_version=load(folder/'code_fingerprints.json'),fingerprint=manifest.get('fingerprint',manifest.get('source_sha256')),
        split_identifier=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest() if splits else None,
        imported=load(folder/'import_record.json'),training_regime={k:config.get(k) for k in ('benign_only_training','training_node_sampling','reference_sampling','checkpoint_selection','label_policy')})
