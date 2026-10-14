import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
import numpy as np
import torch
from ids.theia.data import prepare, validate, windows, records, uid
from ids.theia.metrics import evaluate, select_threshold
from ids.theia.models import FlashNode, MagicNode
from ids.theia.download import BudgetReader
from theia_fixture import make_fixture, node_id, PREFIX

ROOT = Path(__file__).resolve().parents[1]


def config():
    result=json.loads((ROOT/'configs/theia_smoke.json').read_text())
    return result|dict(dataset='synthetic_cdm18',split_mode='explicit',start_ns=0,train_end_ns=1000,validation_end_ns=2000,end_ns=3000,
                      window_ns=1000,max_nodes_per_window=16,max_edges_per_window=24,max_windows_per_partition=20)


class TheiaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.raw,self.labels=make_fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def prepared(self,name='processed',cfg=None,curated=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare([self.raw],self.root/name,cfg or config(),curated_path=self.labels if curated else None)
    def test_real_uuid_canonicalization(self):
        self.assertEqual(uid({'com.bbn.tc.schema.avro.cdm18.UUID':'80370c6e-7a87-ffff-ffff-000000000040'}),'80370C6E-7A87-FFFF-FFFF-000000000040')
        self.assertIsNone(uid(None))
        with self.assertRaises(ValueError):uid('not-a-uuid')
    def test_alignment_chronology_limits_and_direction(self):
        path=self.prepared();self.assertTrue(validate(path)['verified'])
        m=json.loads((path/'manifest.json').read_text());self.assertEqual(m['counts']['missing_endpoint_edges'],0)
        all_ids=[]
        for part in ('train','validation','test'):
            ids=[]
            for entry,g in windows(path,part):
                ids.extend(g['uuids'][g['target']]);self.assertLessEqual(len(g['x']),16)
                self.assertLessEqual(len(g['src']),24)
                self.assertTrue((np.diff(g['timestamps'])>=0).all())
            all_ids.extend(ids);self.assertEqual(len(ids),48)
        self.assertEqual(len(all_ids),len(set(all_ids)))
        db=sqlite3.connect(path/'events.sqlite')
        src,dst=db.execute("SELECT src,dst FROM events WHERE type='EVENT_READ' ORDER BY ts LIMIT 1").fetchone()
        self.assertEqual((src,dst),(node_id(4),node_id(3)));db.close()
    def test_unknown_not_negative_and_policy_explicit(self):
        p=self.prepared(curated=False)
        for _,g in windows(p,'test'):self.assertTrue((g['y']==-1).all())
        with self.assertRaises(ValueError):self.prepared('bad',config()|{'label_policy':'scoped_background'},False)
        p=self.prepared('background',config()|{'label_policy':'scoped_background','background_rationale':'TEST ONLY'},False)
        for _,g in windows(p,'test'):self.assertTrue((g['y']==0).all())
    def test_future_events_and_labels_do_not_change_training_features(self):
        p=self.prepared(); before=[g for _,g in windows(p,'train')]
        lines=self.raw.read_text().splitlines()
        for k,line in enumerate(lines):
            r=json.loads(line);e=r['datum'].get(PREFIX+'Event')
            if e and e['timestampNanos']>=2000:e['type']='EVENT_FUTURE_ONLY';lines[k]=json.dumps(r)
        self.raw.write_text('\n'.join(lines)+'\n')
        p2=self.prepared('changed',curated=False);after=[g for _,g in windows(p2,'train')]
        for a,b in zip(before,after):
            for key in ('x','src','dst','types','target'):np.testing.assert_array_equal(a[key],b[key])
        m=json.loads((p2/'manifest.json').read_text());self.assertNotIn('EVENT_FUTURE_ONLY',m['edge_vocabulary'])
    def test_repeated_uuid_targets_only_once(self):
        r={'datum':{PREFIX+'Event':{'uuid':node_id(99000000),'subject':{PREFIX+'UUID':node_id(1)},'predicateObject':{PREFIX+'UUID':node_id(2)},'timestampNanos':1500,'type':'EVENT_READ'}}}
        with open(self.raw,'a') as f:f.write(json.dumps(r)+'\n')
        p=self.prepared();validate(p)
        with open(p/'targets.csv') as f:rows=list(csv.DictReader(f))
        self.assertEqual(sum(r['uuid']==node_id(1) for r in rows),1)
    def test_duplicate_events_and_two_predicates(self):
        row=json.loads(self.raw.read_text().splitlines()[0]);e=row['datum'][PREFIX+'Event']
        with open(self.raw,'a') as f:f.write(json.dumps(row)+'\n')
        p=self.prepared();m=json.loads((p/'manifest.json').read_text());self.assertEqual(m['counts']['duplicate_edges'],1)
        e['uuid']=node_id(90000001);e['predicateObject2']={PREFIX+'UUID':node_id(1)}
        with open(self.raw,'a') as f:f.write(json.dumps(row)+'\n')
        p=self.prepared('second');db=sqlite3.connect(p/'events.sqlite')
        self.assertEqual(db.execute('SELECT COUNT(*) FROM events WHERE event_id=?',(node_id(90000001),)).fetchone()[0],2);db.close()
    def test_integrity_tamper_detected(self):
        p=self.prepared();f=next(p.glob('*.npz'));f.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'fingerprint'):validate(p)
    def test_stream_formats_and_record_budget(self):
        expected=len(self.raw.read_text().splitlines());gz=self.root/'raw.json.gz'
        with gzip.open(gz,'wb') as f:f.write(self.raw.read_bytes())
        archive=self.root/'raw.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:tar.add(self.raw,arcname='../../never-extracted.json')
        for path in (gz,archive):
            with records(path,10000) as rows:self.assertEqual(sum(1 for _ in rows),expected)
        with self.assertRaises(ValueError):
            with records(self.raw,10) as rows:list(rows)
        reader=BudgetReader(io.BytesIO(b'12345'),3)
        with self.assertRaises(RuntimeError):reader.read(5)
    def test_threshold_and_single_class_metrics(self):
        t,rule=select_threshold([-1,0,0,1,1],[100,.1,.2,.8,.9]);self.assertEqual(t,.8)
        m=evaluate([-1,0,0,1,1],[100,.1,.85,.7,.9],t)
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(1,1,1,1));self.assertEqual(m['unknown_targets'],1)
        self.assertIsNone(select_threshold([-1,1],[.1,.9])[0])
        m=evaluate([0,0],[.1,.2],.5);self.assertIsNone(m['roc_auc']);self.assertIsNone(m['recall']);self.assertFalse(m['binary_metrics_defined'])
    def test_training_and_scores_ignore_ground_truth_labels(self):
        from ids.theia.worker import run
        from sklearn.preprocessing import StandardScaler
        p=self.prepared('labelled'); q=self.prepared('unknown',curated=False)
        for model in ('flash_node','magic_node','random_forest_node'):
            outputs=[]
            for label,data in (('a',p),('b',q)):
                out=self.root/(model+label);out.mkdir()
                cfg=config()|{'epochs':1,'rf_trees':4}
                (out/'config.json').write_text(json.dumps(cfg))
                with contextlib.redirect_stdout(io.StringIO()):run(data,out,model,5)
                with open(out/f'{model}_5_predictions.csv') as stream:rows=list(csv.DictReader(stream))
                outputs.append([(r['uuid'],r['score']) for r in rows])
                training=json.loads((out/f'{model}_5_training.json').read_text())
                scaler=StandardScaler()
                for _,g in windows(data,'train'):scaler.partial_fit(g['x'])
                np.testing.assert_array_equal(training['scaler_mean'],scaler.mean_)
                if label=='b':
                    self.assertIsNone(json.loads((out/f'{model}_5_threshold.json').read_text())['threshold'])
            self.assertEqual(outputs[0],outputs[1])

    def test_missing_definitions_and_import_caps_reported(self):
        from ids.theia.data import import_raw
        cfg=config()|{'max_import_nodes':2,'max_import_edges':10}
        db,counts,_=import_raw([self.raw],self.root/'limited.sqlite',cfg)
        try:
            self.assertEqual(counts['entities_imported'],2)
            self.assertEqual(counts['edges_imported'],10)
            self.assertGreater(counts['entities_over_limit'],0)
            self.assertGreater(counts['edges_over_limit'],0)
            self.assertGreater(counts['missing_endpoint_edges'],0)
        finally:db.close()

    def test_label_conflicts_and_bad_records_fail_loudly(self):
        from ids.theia.data import read_labels,import_raw
        positives=self.root/'positives.txt';positives.write_text(node_id(3)+'\n')
        with self.assertRaisesRegex(ValueError,'Conflicting label'):read_labels(positives,self.labels)
        with open(self.raw,'a') as f:f.write('{malformed}\n')
        with self.assertRaisesRegex(ValueError,'SYNTHETIC_cdm18'):import_raw([self.raw],self.root/'invalid.sqlite',config())

    def test_node_equivariance_gradients_and_checkpoint(self):
        torch.manual_seed(10);x=torch.randn(5,8);s=torch.tensor([0,1,2,3]);d=torch.tensor([1,2,3,4]);perm=torch.tensor([3,1,0,4,2]);inv=torch.argsort(perm)
        for model in (FlashNode(8,16,4),MagicNode(8,16)):
            fn=model if isinstance(model,FlashNode) else model.encode
            a=fn(x,s,d);b=fn(x[perm],inv[s],inv[d]);self.assertTrue(torch.allclose(a[perm],b,atol=1e-6));self.assertEqual(len(a),5)
            loss=a.square().mean() if isinstance(model,FlashNode) else model.loss(x,s,d,.5)
            loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            path=self.root/'checkpoint.pt';torch.save(model.state_dict(),path)
            reloaded=FlashNode(8,16,4) if isinstance(model,FlashNode) else MagicNode(8,16)
            reloaded.load_state_dict(torch.load(path,weights_only=True))
            fn2=reloaded if isinstance(model,FlashNode) else reloaded.encode
            self.assertTrue(torch.equal(a,fn2(x,s,d)))

if __name__=='__main__':unittest.main()
