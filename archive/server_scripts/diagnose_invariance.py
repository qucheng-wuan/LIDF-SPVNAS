import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/spvnas')]
from launch import check_idle
import argparse
p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--variant',default='original');p.add_argument('--single',action='store_true');args=p.parse_args()
check_idle([args.gpu])
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUDA_HOME='/home/quchengzou/miniconda3/envs/lidf',TORCH_EXTENSIONS_DIR=str(ROOT/'runs/spvnas_fusion/extensions'),PATH='/home/quchengzou/miniconda3/envs/lidf/bin:'+os.environ['PATH'])
import torch,numpy as np
from entry import options
from torch.utils.data import DataLoader
from datasets.cleargrasp_synthetic_dataset import get_dataset
os.chdir(ROOT/'src')
from models.pipeline import LIDF
opt=options([ROOT/f'src/experiments/spvnas_fusion/{args.variant}.yaml']);opt.gpu_id=0;opt.dist.ddp=False
opt.base_log_dir=str(ROOT/'runs/spvnas_fusion/diagnostic'/args.variant)
opt.grid.valid_sample_num=-1;opt.grid.miss_sample_num=-1
torch.manual_seed(opt.seed);np.random.seed(opt.seed)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
model=LIDF(opt,torch.device('cuda:0')).eval()
if args.variant != 'original':
 import core.models.utils as sparse_utils
 original_query=sparse_utils.sphashquery
 def checked_query(query,target,kernel_size=1):
  result=original_query(query,target,kernel_size)
  if kernel_size==1 and (result<0).any():
   print('UNMATCHED_POINT_TO_VOXEL',int((result<0).sum()),'of',len(query),'target first batch',int(target[0,0]),flush=True)
  return result
 sparse_utils.sphashquery=checked_query

batch=next(iter(DataLoader(get_dataset(opt.dataset.cleargrasp_root_dir,opt.dataset.dict.copy(),'test'),batch_size=2)))
flip={k:v.flip(0) if torch.is_tensor(v) else v[::-1] for k,v in batch.items()}
with torch.no_grad():
 _,a,_=model(batch,'valid',0);_,b,_=model(flip,'valid',0)
 for bid in range(2):
  if args.single:
   single={k:v[bid:bid+1] for k,v in batch.items()}
   _,b,_=model(single,'valid',0)
  target_bid=0 if args.single else 1-bid
  x=a['pred_pos'][a['miss_bid']==bid];y=b['pred_pos'][b['miss_bid']==target_bid]
  diff=(x-y).abs().max(1).values
  print('sample',bid,'maxdiff',diff.max().item(),'changed',(diff>1e-5).sum().item())
  xa=a['intersect_rgb_feat'][a['miss_bid'][a['miss_ray_intersect_idx']]==bid]
  xb=b['intersect_rgb_feat'][b['miss_bid'][b['miss_ray_intersect_idx']]==target_bid]
  print('rgb feat maxdiff',(xa-xb).abs().max().item())
  va=a['intersect_voxel_feat'][a['miss_bid'][a['miss_ray_intersect_idx']]==bid]
  vb=b['intersect_voxel_feat'][b['miss_bid'][b['miss_ray_intersect_idx']]==target_bid]
  print('voxel feat maxdiff',(va-vb).abs().max().item())
  for idx in torch.nonzero(diff>1e-5).reshape(-1)[:4]:
   ra=torch.nonzero(a['miss_bid']==bid).reshape(-1)[idx];rb=torch.nonzero(b['miss_bid']==target_bid).reshape(-1)[idx]
   ma=a['miss_ray_intersect_idx']==ra;mb=b['miss_ray_intersect_idx']==rb
   print('ray',idx.item(),'logits',a['pred_prob_end'][ma].flatten().tolist(),b['pred_prob_end'][mb].flatten().tolist(),'probs',a['pred_prob_end_softmax'][ma].tolist(),b['pred_prob_end_softmax'][mb].tolist())
