import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from study_dashboard_data import enrich
from import_hpc_results import import_run

class DashboardEvidenceTests(unittest.TestCase):
    def test_real_counts_and_compatibility(self):
        p=ROOT/'results/theia/study_full_v1'
        rows=json.loads((p/'metrics.json').read_text())
        e=enrich(p,rows)
        self.assertEqual(e['dataset_counts']['original_node_occurrences'],1623966)
        self.assertEqual(e['dataset_counts']['processed_edges'],2874821)
        self.assertIsNone(e['dataset_counts']['unique_entities'])
        self.assertIsNone(e['dataset_counts']['raw_records'])
        self.assertEqual(sum(p['targets'] for p in e['partitions']),1623966)
        self.assertTrue(e['compatible'])
        self.assertFalse(e['subset'])
        self.assertEqual(e['partitions'][2]['positive'],25319)
        rows.pop()
        self.assertFalse(enrich(p,rows)['compatible'])
    def test_f2_and_throughput(self):
        with tempfile.TemporaryDirectory() as t:
            rows=[dict(model='test',seed=1,tp=3,fp=2,fn=1,test_targets=10,inference_seconds=2)]
            enrich(Path(t),rows)
            self.assertAlmostEqual(rows[0]['f2'],15/21)
            self.assertEqual(rows[0]['throughput'],5)
    def test_subset(self):
        p=ROOT/'results/theia/study_smoke_v2'
        e=enrich(p,json.loads((p/'metrics.json').read_text()))
        self.assertTrue(e['subset'])
        self.assertEqual(e['dataset_counts']['processed_node_occurrences'],20480)
    def test_missing_and_explicit_states(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for state in ['queued','running','failed','interrupted']:
                (p/'run_status.json').write_text(json.dumps({'state':state}))
                e=enrich(p,[])
                self.assertEqual(e['run_state'],state)
                self.assertFalse(e['compatible'])
                self.assertIsNone(e['dataset_counts']['processed_edges'])
    def test_import_preserves_snapshot_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'source';src.mkdir()
            (src/'config.json').write_text('{}')
            (src/'run_status.json').write_text('{"state":"interrupted"}')
            out=import_run(src,'snapshot',root)
            self.assertEqual((out/'config.json').read_text(),'{}')
            self.assertEqual(json.loads((out/'import_record.json').read_text())['state'],'interrupted')
            with self.assertRaises(ValueError):import_run(src,'snapshot',root)
            with self.assertRaises(ValueError):import_run(src,'../escape',root)
    def test_streamspot_counts(self):
        p=ROOT/'results/quick_20261013T063017021814Z'
        e=enrich(p,json.loads((p/'metrics.json').read_text()))
        self.assertEqual(e['dataset_counts']['graphs'],600)
        self.assertEqual(e['dataset_counts']['raw_records'],89770902)
        self.assertEqual(e['dataset_counts']['processed_edges'],307200)
        self.assertTrue(e['subset'])
        self.assertEqual(sum(x['targets'] for x in e['partitions']),600)
        self.assertFalse(e['compatible'])
if __name__=='__main__':unittest.main()
