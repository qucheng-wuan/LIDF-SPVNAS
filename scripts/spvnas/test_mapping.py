"""CPU checks of real quantization and LIDF voxel mapping; no mock backbone."""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from models.spvnas_fusion import quantize, mean_pool, mean_pool_matched
from utils.point_utils import batch_get_occupied_idx


class MappingTests(unittest.TestCase):
    def test_fast_quantization_matches_checked_path(self):
        torch.manual_seed(2026)
        xyz = torch.rand(600, 3)
        batch = torch.arange(3).repeat_interleave(200)
        checked = quantize(xyz, batch, .05, check=True)
        fast = quantize(xyz, batch, .05, check=False)
        for a, b in zip(checked, fast):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_sparse_matches_do_not_contaminate_voxel_zero(self):
        features=torch.tensor([[1.,2.],[1000.,2000.],[3.,4.]],requires_grad=True)
        indices=torch.tensor([0,-1,1])
        pooled=mean_pool_matched(features,indices,3)
        torch.testing.assert_close(pooled,torch.tensor([[1.,2.],[3.,4.],[0.,0.]]))
        pooled.sum().backward()
        torch.testing.assert_close(features.grad,torch.tensor([[1.,1.],[0.,0.],[1.,1.]]))

    def test_duplicates_and_batch_isolation(self):
        xyz=torch.tensor([[.01,.02,.7],[.011,.021,.701],[.3,.1,.9]]).repeat(2,1)
        batch=torch.tensor([0,0,0,1,1,1])
        _,coords,inverse=quantize(xyz,batch,.05)
        self.assertEqual(len(coords),4)
        self.assertEqual(inverse[0],inverse[1])
        self.assertNotEqual(inverse[0],inverse[3])
        feat=torch.arange(12,dtype=torch.float32).reshape(6,2).requires_grad_()
        pooled=mean_pool(feat,inverse,len(coords))
        restored=pooled[inverse]
        torch.testing.assert_close(restored[0],feat[:2].mean(0))
        torch.testing.assert_close(restored[3],feat[3:5].mean(0))
        restored.square().sum().backward()
        self.assertTrue((feat.grad.abs().sum(1)>0).all())

    def test_lidf_cropping_pooling_and_permutation(self):
        torch.manual_seed(7)
        xyz=torch.rand(200,3)*.9+.05
        xyz[0]=torch.tensor([8.,8.,8.])
        batch=torch.arange(2).repeat_interleave(100)
        feat=torch.randn(200,5)
        vox,rev,pids,_,_=batch_get_occupied_idx(xyz,batch[:,None],crop_size=.2)
        self.assertFalse((pids==0).any())
        self.assertTrue(torch.equal(batch[pids],vox[rev,0]))
        pooled=mean_pool(feat[pids],rev,len(vox))
        for k in range(len(vox)):
            torch.testing.assert_close(pooled[k],feat[pids[rev==k]].mean(0))
        order=torch.randperm(200)
        vox2,rev2,pids2,_,_=batch_get_occupied_idx(xyz[order],batch[order,None],crop_size=.2)
        self.assertTrue(torch.equal(vox,vox2))
        torch.testing.assert_close(pooled,mean_pool(feat[order][pids2],rev2,len(vox2)))
        _,coords,inv=quantize(xyz,batch,.01)
        _,coords2,inv2=quantize(xyz,1-batch,.01)
        # Label permutation retains the physical per-sample grid exactly.
        self.assertTrue(torch.equal(coords[inv,1:],coords2[inv2,1:]))

if __name__=='__main__':
    unittest.main()
