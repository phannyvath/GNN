"""Protocol and security checks for the static graph importer."""
import io
import pickle
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ids.theia.processed import GraphReader, features, scheme_attribute, Record, prepare, validate_processed


class ProcessedTests(unittest.TestCase):
    def test_pickle_rejects_executable_globals(self):
        import os
        with self.assertRaises(pickle.UnpicklingError):
            GraphReader(io.BytesIO(pickle.dumps(os.system))).load()

    def test_pickle_rejects_attribute_escape(self):
        with self.assertRaises(pickle.UnpicklingError):
            scheme_attribute(Record, '__subclasses__')

    def test_features_direction_and_unknown_type(self):
        src = np.array([0, 0, 1]); dst = np.array([1, 2, 2]); et = np.array([7, 8, 7])
        x = features(src, dst, et, 3, {7: 1})
        np.testing.assert_allclose(np.expm1(x), [[1,1,0,0], [0,1,0,1], [0,0,1,1]], atol=1e-6)

    def test_isolated_nodes_preserved(self):
        x = features(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), 2, {7: 1})
        self.assertEqual(x.shape, (2,4))
        self.assertTrue((x == 0).all())

    def test_static_split_label_alignment_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir()
            for name in ['train0.pkl','train1.pkl','train2.pkl','train3.pkl','test0.pkl']:
                (source/name).write_bytes(b'fixture replaced by mocked graph reader')
            (source/'metadata.json').write_text(json.dumps(dict(n_train=4, n_test=1,
                node_feature_dim=2, edge_feature_dim=13, malicious=[[1],['NA']])))
            # Include keys above 9 to catch JSON integer-key fingerprint ordering.
            graph = (np.arange(12), np.arange(1,13), np.arange(13)%2, np.arange(12), 2, 13)
            with patch('ids.theia.processed.read_graph', return_value=graph):
                data = prepare(source, root/'prepared', {'label_policy':'upstream_positive_presumed_background'})
            audit = validate_processed(data)
            self.assertEqual(audit['targets'], {'train':39, 'validation':13, 'test':13})
            with np.load(data/'test_4.npz') as test:
                self.assertEqual(np.flatnonzero(test['y']).tolist(), [1])
                self.assertEqual(test['uuids'][1], 'test0.pkl:node:1')
            with np.load(data/'validation_3.npz') as val:
                self.assertTrue((val['y'] == 0).all())
            with np.load(data/'train_0.npz') as train:
                self.assertTrue((train['y'] == -1).all())
            (data/'targets.csv').write_text('tampered')
            with self.assertRaises(ValueError):
                validate_processed(data)


if __name__ == '__main__':
    unittest.main()
