from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from models.ray_selection import stable_ray_argmax

class RaySelectionTests(unittest.TestCase):
    def test_equal_maxima_negative_scores_and_empty_ray(self):
        scores=torch.tensor([.3,.3,-2.,-1.,.5,.1])
        rays=torch.tensor([0,0,1,1,2,2])
        self.assertEqual(stable_ray_argmax(scores,rays,4).tolist(),[0,3,4,6])
    def test_batch_reordering(self):
        scores=torch.tensor([.5,.5,.3,.2,.2,.1])
        rays=torch.tensor([0,0,1,2,2,3])
        order=torch.tensor([3,4,5,0,1,2])
        selected=stable_ray_argmax(scores,rays,4)
        flipped=stable_ray_argmax(scores[order],(rays[order]+2)%4,4)
        torch.testing.assert_close(order[flipped][torch.tensor([2,3,0,1])],selected)

if __name__=='__main__':unittest.main()
