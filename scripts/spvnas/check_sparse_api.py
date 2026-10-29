"""Real SparseTensor/BN CPU checks; CUDA queried by import, no GPU computation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/spvnas')]
import torch
from torch import nn
import torchsparse
from torchsparse import SparseTensor
from torchsparse.nn import BatchNorm
from models.spvnas_fusion import convert_sparse_sync_batchnorm,SparseSyncBatchNorm
from core.models.semantic_kitti.spvnas import SPVNAS

record={'torchsparse':torchsparse.__version__,'cpu_bn':{}}
features=torch.randn(7,3,requires_grad=True)
coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,1,0,0]],dtype=torch.int32)
value=SparseTensor(feats=features,coords=coords)
module=nn.Sequential(BatchNorm(3)).eval()
reference=module(value).F.detach().clone()
parameters=list(module.parameters())
convert_sparse_sync_batchnorm(module)
assert isinstance(module[0],SparseSyncBatchNorm)
assert list(module.parameters())[0] is parameters[0]
result=module(value)
assert torch.equal(result.C,coords)
assert result._caches is value._caches
torch.testing.assert_close(reference,result.F)
result.F.square().sum().backward()
assert features.grad.abs().sum()>0
record['cpu_bn']={'conversion':'passed','feature_equivalence':'passed','coordinate_cache_preservation':'passed','input_gradient':'passed'}
for size in [20,65]:
 name=f'SemanticKITTI_val_SPVNAS@{size}GMACs'
 config=json.loads((ROOT/'configs/spvnas_models'/f'{size}g.json').read_text())
 model=SPVNAS(config['num_classes'],macro_depth_constraint=1,pres=config['pres'],vres=config['vres'])
 model.manual_select(config)
 assert model.cur_outputs_channels==config['output_channels']
 record[name]={'supernet_cpu_constructor':'passed','manual_select':'passed','determinize_strict_load':'not_run_requires_idle_gpu'}
print(json.dumps(record,indent=2))
(ROOT/'runs/spvnas_fusion').mkdir(parents=True,exist_ok=True)
(ROOT/'runs/spvnas_fusion/sparse_api_checks.json').write_text(json.dumps(record,indent=2))
