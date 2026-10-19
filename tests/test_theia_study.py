"""Study-level contracts: exact reference, isolation, labels, metrics and epoch resume."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from ids.theia.common import fingerprint,save_json,sha256
from ids.theia.metrics import evaluate,select_threshold
from ids.theia.study_data import make_data,verify_data,validate_config
from ids.theia.study_worker import ExactReference,finite,run,atomic_torch,inference
from ids.theia.study_report import aggregate,load_predictions,METRICS

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    c=json.loads((ROOT/'configs/theia_study_full.json').read_text())
    c.update(epochs=2,patience=3,rf_trees=4,hidden=4,threads=1)
    source=root/'source';source.mkdir()
    for name in ['train0.pkl','train1.pkl','train2.pkl','train3.pkl','test0.pkl']:(source/name).write_bytes(name.encode())
    (source/'graphs.zip').write_bytes(b'fixture not an upstream archive')
    save_json(source/'metadata.json',dict(n_train=4,n_test=1,node_feature_dim=2,edge_feature_dim=2,malicious=[[1],['NA']]))
    src=np.array([0,0,1,2,3,4,5,6,7,8,9,10],dtype=np.int64);dst=np.arange(1,13,dtype=np.int64)%13
    graph=(src,dst,np.arange(13,dtype=np.int64)%2,np.arange(12,dtype=np.int64)%2,2,2)
    return c,source,graph


class StudyTests(unittest.TestCase):
    def test_float32_score_csv_threshold_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            g={'x':np.zeros((2,1)), 'y':np.array([0,0]), 'uuids':np.array(['a','b'])}
            with patch('ids.theia.study_worker.batches',return_value=iter([({},g)])),patch('ids.theia.study_worker.score',return_value=np.array([.2,.2],dtype=np.float32)):
                y,s,_=inference('flash_node',None,None,None,Path(tmp),{},'validation',Path(tmp))
            yy,ss,_,_=load_predictions(Path(tmp)/'validation_predictions.csv','validation')
            self.assertEqual(select_threshold(y,s),select_threshold(yy,ss))

    def test_summary_uses_sample_standard_deviation(self):
        rows=[dict(model='flash_node',seed=i,**{k:v for k in METRICS}) for i,v in enumerate([.1,.2,.3])]
        result=next(r for r in aggregate(rows) if r['model']=='flash_node' and r['metric']=='f1')
        self.assertAlmostEqual(result['mean'],.2);self.assertAlmostEqual(result['sample_std'],.1)

    def test_exact_reference_preserves_all_duplicate_occurrences(self):
        z=np.array([[0,0],[0,0],[0,0],[1,1],[2,3],[2,3]],dtype=np.float32)
        u,counts=np.unique(z,axis=0,return_counts=True);q=np.array([[0,0],[1,0],[2,3]],dtype=np.float32)
        actual=ExactReference(u,counts,5,batch=2,threads=1).score(q)
        expected=NearestNeighbors(n_neighbors=5,algorithm='brute').fit(z).kneighbors(q)[0].mean(1)
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_metric_definition_nulls_and_finite_outputs(self):
        m=evaluate([0,0,1,1],[.1,.9,.8,.2],.5)
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1]);self.assertEqual(m['f1'],.5)
        self.assertEqual(m['roc_auc'],.5);self.assertAlmostEqual(m['average_precision'],7/12)
        self.assertIsNone(evaluate([0,0],[.1,.2],.3)['roc_auc'])
        with self.assertRaises(ValueError):finite([np.nan])
        threshold,_=select_threshold([0,0],[.2,.2],.01);self.assertGreater(threshold,.2)

    def test_three_seeds_and_split_contract(self):
        c=json.loads((ROOT/'configs/theia_study_full.json').read_text());validate_config(c)
        c['seeds']=[42]
        with self.assertRaises(ValueError):validate_config(c)
        c['seeds']=[42,43,44];c['train_files'].append('test0.pkl')
        with self.assertRaises(ValueError):validate_config(c)

    def test_label_alignment_even_after_valid_hash_update(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);c,source,graph=fixture(root)
            with patch('ids.theia.processed.read_graph',return_value=graph),patch('ids.theia.study_data.read_graph',return_value=graph):
                make_data(source,root/'data',c);verify_data(root/'data',source,c)
                path=root/'data/test_4.npz'
                with np.load(path) as f:g={k:f[k] for k in f.files}
                g['y'][1]=0;g['y'][2]=1;np.savez_compressed(path,**g)
                m=json.loads((root/'data/manifest.json').read_text());m['shards'][-1]['sha256']=sha256(path);m.pop('fingerprint');m['fingerprint']=fingerprint(m);save_json(root/'data/manifest.json',m)
                with self.assertRaisesRegex(ValueError,'Malicious-index'):verify_data(root/'data',source,c)

    def test_test_changes_do_not_change_training_features(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);c,source,graph=fixture(root)
            with patch('ids.theia.processed.read_graph',return_value=graph),patch('ids.theia.study_data.read_graph',return_value=graph):make_data(source,root/'a',c)
            save_json(source/'metadata.json',dict(n_train=4,n_test=1,node_feature_dim=2,edge_feature_dim=2,malicious=[[2,3],['NA','NA']]))
            def altered(path):
                if Path(path).name=='test0.pkl':return (graph[1],graph[0],graph[2],graph[3],2,2)
                return graph
            with patch('ids.theia.processed.read_graph',side_effect=altered),patch('ids.theia.study_data.read_graph',side_effect=altered):make_data(source,root/'b',c)
            for i in range(3):
                with np.load(root/f'a/train_{i}.npz') as a,np.load(root/f'b/train_{i}.npz') as b:
                    np.testing.assert_array_equal(a['x'],b['x']);np.testing.assert_array_equal(a['types'],b['types'])

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);c,source,graph=fixture(root)
            for name in ['straight','resumed']:
                study=root/name;study.mkdir();save_json(study/'config.json',c)
                with patch('ids.theia.processed.read_graph',return_value=graph),patch('ids.theia.study_data.read_graph',return_value=graph):make_data(source,study/'data',c)
            run(root/'straight','magic_node',42)
            def interrupt(path,obj):
                atomic_torch(path,obj)
                if path.name=='last.pt' and obj['epoch']==1:raise RuntimeError('simulated interruption')
            with patch('ids.theia.study_worker.atomic_torch',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'):run(root/'resumed','magic_node',42)
            run(root/'resumed','magic_node',42,resume=True)
            left=load_predictions(root/'straight/magic_node_42/test_predictions.csv','test')[1]
            right=load_predictions(root/'resumed/magic_node_42/test_predictions.csv','test')[1]
            np.testing.assert_allclose(left,right,atol=1e-7)
            a=torch.load(root/'straight/magic_node_42/last.pt',weights_only=True)
            b=torch.load(root/'resumed/magic_node_42/last.pt',weights_only=True)
            for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
