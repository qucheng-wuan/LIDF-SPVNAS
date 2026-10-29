"""CPU construction, strict legacy checkpoint check, and dataset loading."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/spvnas')]
from entry import options
import torch
from torch.utils.data import DataLoader
os.chdir(ROOT/'src')
from models.pipeline import LIDF
from trainers.train_spvnas import load_base,parameter_groups
from datasets.cleargrasp_synthetic_dataset import get_dataset
opt=options([ROOT/'src/experiments/spvnas_fusion/original.yaml'])
opt.gpu_id=0
opt.dist.ddp=False
model=LIDF(opt,torch.device('cpu'))
assert 'torchsparse' not in sys.modules
record={'original_cpu_build':'passed','original_lazy_import':'passed','parameters':sum(p.numel() for p in model.parameters())}
# Strict check against actual modules, including all buffers and decoder dimensions.
frozen=options([ROOT/'src/experiments/spvnas_fusion/spvnas_fusion_20g.yaml',ROOT/'src/experiments/spvnas_fusion/freeze_20g.yaml'])
ckpt=torch.load(frozen.checkpoint_path,map_location='cpu')
load_base(model,ckpt)
record['legacy_checkpoint']={'path':frozen.checkpoint_path,'epoch':ckpt.get('epoch'),'strict_base_load':'passed'}
record['groups']=[{'name':g['name'],'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in parameter_groups(model,opt)]
for split in ['train','test']:
 params=opt.dataset.dict.copy(); params['use_data_augmentation']=False
 dataset=get_dataset(opt.dataset.cleargrasp_root_dir,params,split,'known')
 batch=next(iter(DataLoader(dataset,batch_size=1)))
 for key in ['rgb','xyz','depth','xyz_corrupt','depth_corrupt']:
  assert torch.isfinite(batch[key]).all(),key
 record[split]={'count':len(dataset),'shapes':{k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}}
(ROOT/'runs/spvnas_fusion/cpu_checks.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
