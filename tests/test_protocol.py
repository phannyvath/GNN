import unittest
import numpy as np
import torch
from ids.data import partitions
from ids.metrics import evaluate,threshold_from_validation
from ids.models import mean_neighbors,GraphAttention,FlashInspired,MagicInspired
from ids.baselines import shingle_features,summary_features

class ProtocolTests(unittest.TestCase):
    def test_splits_disjoint_scenario_counts(self):
        splits=partitions(range(600),2026)
        self.assertEqual([len(splits[k]) for k in ['train','validation','test']],[360,120,120])
        self.assertEqual(len(set(sum(splits.values(),[]))),600)
        for k,n in [('train',60),('validation',20),('test',20)]:
            for scene in range(6): self.assertEqual(sum(i//100==scene for i in splits[k]),n)
        self.assertEqual(splits,partitions(range(600),2026))
    def test_partial_dataset_rejected(self):
        with self.assertRaises(ValueError): partitions(range(599),1)
    def test_direction(self):
        x=torch.tensor([[2.],[8.],[5.]])
        out=mean_neighbors(x,torch.tensor([0]),torch.tensor([1]))
        self.assertTrue(torch.equal(out,torch.tensor([[0.],[2.],[0.]])))
    def test_threshold_and_confusion(self):
        t=threshold_from_validation([0,0,1,1],[.1,.2,.8,.9])
        m=evaluate([0,0,1,1],[.1,.85,.7,.9],t)
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(1,1,1,1))
        self.assertEqual(m['f1'],.5)
    def test_no_id_or_label_features(self):
        g={'id':1,'y':0,'x':np.eye(3,dtype=np.float32),'src':np.array([0,1]),'dst':np.array([1,2]),'types':[1,2,3],'edge_types':[1,2]}
        changed=dict(g,id=399,y=1)
        self.assertTrue(np.array_equal(summary_features(g),summary_features(changed)))
        self.assertTrue(np.array_equal(shingle_features(g),shingle_features(changed)))
    def test_model_gradient_and_permutation_invariance(self):
        torch.manual_seed(3); x=torch.randn(4,5); src=torch.tensor([0,1,2]); dst=torch.tensor([1,2,3])
        perm=torch.tensor([2,0,3,1]); inverse=torch.argsort(perm)
        for model in [FlashInspired(5,8),MagicInspired(5,8)]:
            model.eval()
            fn=model if isinstance(model,FlashInspired) else model.embedding
            a=fn(x,src,dst); b=fn(x[perm],inverse[src],inverse[dst])
            self.assertTrue(torch.allclose(a,b,atol=1e-6))
            loss=a.sum() if isinstance(model,FlashInspired) else model.loss(x,src,dst,.4)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_attention_with_no_edges(self):
        m=GraphAttention(3,5); z=m(torch.ones(2,3),torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))
        self.assertEqual(z.shape,(2,5)); self.assertTrue(torch.isfinite(z).all())
if __name__=='__main__': unittest.main()
