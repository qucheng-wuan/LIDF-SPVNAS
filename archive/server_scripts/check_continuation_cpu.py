"""CPU-only verification; never initializes CUDA or interrupts production."""
import ast
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/spvnas'))
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from entry import options
from datasets import cleargrasp_synthetic_dataset as old_data
from datasets import cleargrasp_fast as fast_data
from utils import data_augmentation
from utils.training_resume import (capture_runtime, restore_runtime, legacy_runtime,
    validate_checkpoint, parameter_layout, training_protocol)

torch.set_num_threads(1)
cv2.setNumThreads(1)
PARENT = OUT.parent
opt = options([ROOT/'src/experiments/spvnas_fusion/spvnas_fusion_20g.yaml',
               PARENT/'configs/common.yaml'])
result = dict(cuda_initialized_before=torch.cuda.is_initialized())
old = old_data.get_dataset('/ssd/polyu', opt.dataset.dict.copy(), 'train')
fast = fast_data.get_dataset('/ssd/polyu', opt.dataset.dict.copy(), 'train')
assert old.image_paths == fast.image_paths
protocol = training_protocol(opt, old)
(OUT/'provenance/parent_training_protocol.json').write_text(json.dumps(protocol, indent=2))
generator = torch.Generator().manual_seed(2026)
def capture():
    return capture_runtime(generator)
def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b

# Full samples include RGB augmentation, ellipse masks and camera transformations.
indices = [0,1,17,100,1000,40907,45453,2026,9876,23456,34567,43210]
times = {'old': [], 'fast': []}
for i in indices:
    random.seed(2000+i); np.random.seed(2000+i); torch.manual_seed(2000+i)
    state = capture()
    t=time.perf_counter(); a=old[i]; times['old'].append(time.perf_counter()-t)
    after=capture()
    restore_runtime(state,generator)
    t=time.perf_counter(); b=fast[i]; times['fast'].append(time.perf_counter()-t)
    assert equal(a,b), 'Preprocessing differs for sample '+str(i)
    assert equal(after,capture()), 'RNG consumption changed for '+str(i)
result['full_sample_equivalence'] = dict(samples=len(indices), indices=indices,
    all_fields_bitwise_equal=True, rng_equal=True, timing_seconds=times,
    caveat='CPU diagnostic timings; old reads first, cache/host-load effects remain')

# Empty background, all foreground, disconnected components, RGB masks.
masks=[np.zeros((11,13),np.uint8),np.full((11,13),255,np.uint8),
       np.eye(13,dtype=np.uint8)*255,
       np.random.RandomState(4).choice([0,128,255],size=(41,37)).astype(np.uint8)]
for mask in masks + [np.repeat(x[...,None],3,axis=2) for x in masks]:
    assert np.array_equal(old.process_label(mask), fast.process_label(mask))
result['label_boundary_cases']=8
for h,w in [(11,13),(240,320)]:
    camera=dict(fx=200.,fy=199.,cx=w/2+.1,cy=h/2+.2,yres=h,xres=w)
    depth=np.random.RandomState(3).rand(h,w).astype(np.float32)
    assert np.array_equal(data_augmentation.compute_xyz(depth,camera),
                          fast_data.compute_xyz_cached(depth,camera))
    assert not fast_data.pixel_indices(h,w).flags.writeable
result['cached_xyz_bitwise_equal']=True

# Actual legacy checkpoint: complete named CPU parameter shell checks all
# parameter/group shapes, counters and Adam storage; not a CUDA model forward.
path=PARENT/'spvnas_fusion_20g/ckpt/stage1/latest_network.pth'
ckpt=torch.load(path,map_location='cpu')
shell=nn.Module()
buffer_suffixes=('running_mean','running_var','num_batches_tracked')
for name,value in ckpt['model'].items():
    current=shell
    parts=name.split('.')
    for part in parts[:-1]:
        if part not in current._modules: current.add_module(part, nn.Module())
        current=current._modules[part]
    if parts[-1] in buffer_suffixes: current.register_buffer(parts[-1],value.clone())
    else: current.register_parameter(parts[-1],nn.Parameter(value.clone()))
named=dict(shell.named_parameters())
prefixes={'lidf':('resnet_model.','pnet_model.','offset_dec.','prob_dec.'),
          'spvnas':('spvnas_branch.backbone.',),
          'adapters':('spvnas_branch.input_adapter.','spvnas_branch.fusion.')}
groups=[]
for group in ckpt['optimizer']['param_groups']:
    names=[n for prefix in prefixes[group['name']] for n in named if n.startswith(prefix)]
    groups.append(dict(name=group['name'],params=[named[n] for n in names],lr=group['initial_lr']))
optimizer=torch.optim.Adam(groups)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,60,gamma=.1)
layout=validate_checkpoint(ckpt,shell,optimizer,opt.model.variant,60)
shell.load_state_dict(ckpt['model'],strict=True)
optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
assert equal(optimizer.state_dict(),ckpt['optimizer'])
assert equal(scheduler.state_dict(),ckpt['scheduler'])
negative=[]
for field,value in [('variant','original'),('epoch',0),('completed_epochs',61)]:
    bad=dict(ckpt);bad[field]=value
    try: validate_checkpoint(bad,shell,optimizer,opt.model.variant,60)
    except ValueError: negative.append(field)
    else: raise AssertionError('Accepted bad '+field)
bad=dict(ckpt);bad['parameter_layout']=list(reversed(layout))
try: validate_checkpoint(bad,shell,optimizer,opt.model.variant,60)
except ValueError: negative.append('parameter_layout')
else: raise AssertionError('Accepted reordered layout')
result['actual_checkpoint']=dict(path=str(path), completed_epochs=ckpt['completed_epochs'],
    global_step=ckpt['global_step'], strict_model_load=True, parameter_count=len(named),
    optimizer_scheduler_restored_exactly=True, rejected_mismatches=negative)
del shell,named,groups,optimizer,scheduler,ckpt

# Save/reload future RNG + BN + Adam + scheduler, compare subsequent updates.
def new_toy():
    model=nn.Sequential(nn.Linear(5,7),nn.BatchNorm1d(7),nn.Tanh(),nn.Linear(7,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,3,gamma=.5)
    return model,optimizer,scheduler
def advance(model,optimizer,scheduler):
    x=torch.randn(6,5)*(1+np.random.rand())+random.random()
    loss=model(x).square().mean()
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step()
    return float(loss)
random.seed(9);np.random.seed(9);torch.manual_seed(9)
m,o,s=new_toy()
for _ in range(3):advance(m,o,s)
state=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),rng=capture()))
losses=[advance(m,o,s) for _ in range(3)]
m2,o2,s2=new_toy()
m2.load_state_dict(state['model']);o2.load_state_dict(state['optimizer']);s2.load_state_dict(state['scheduler'])
restore_runtime(state['rng'],generator)
losses2=[advance(m2,o2,s2) for _ in range(3)]
assert losses==losses2 and equal(m.state_dict(),m2.state_dict()) and equal(o.state_dict(),o2.state_dict()) and equal(s.state_dict(),s2.state_dict())
result['cpu_continuous_vs_resume']=dict(steps_before=3,steps_after=3,bitwise_equal=True,includes=['BN','Adam','StepLR','Python RNG','NumPy RNG','Torch CPU RNG','loader generator'])

# Prove one generator base-seed draw per iterator in the actual installed loader.
dataset=TensorDataset(torch.arange(16))
for rank in range(4):
    g=torch.Generator().manual_seed(2026+rank)
    loader=DataLoader(dataset,batch_size=2,num_workers=0,
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=2026),generator=g)
    for epoch in range(8):
        loader.sampler.set_epoch(epoch);list(loader)
    g2=torch.Generator()
    legacy_runtime(2026,rank,8,g2)
    assert torch.equal(g.get_state(),g2.get_state())
result['legacy_loader_generator_reconstruction']=dict(ranks=4,epochs=8,bitwise_equal=True,
    limitation='Main-process Python/NumPy/Torch/CUDA RNG still unavailable from old checkpoint')
for path in [ROOT/'src/trainers/train_spvnas.py',ROOT/'src/utils/training_resume.py',ROOT/'src/datasets/cleargrasp_fast.py']:
    ast.parse(path.read_text())
# Execute the ACTUAL train_epoch method on a small CPU-only model. This is
# control-flow testing, not a real LIDF epoch or an additional training run.
from contextlib import nullcontext
from types import SimpleNamespace as NS
trainer_ast=ast.parse((ROOT/'src/trainers/train_spvnas.py').read_text())
cls=next(n for n in trainer_ast.body if isinstance(n,ast.ClassDef) and n.name=='SPVNASExperiment')
ns=dict(TrainLIDF=object,torch=torch,np=np,json=json,time=time,Path=Path,
        nullcontext=nullcontext,parameter_layout=parameter_layout,
        BASE_MODULES=('resnet_model','pnet_model','offset_dec','prob_dec'))
exec(compile(ast.Module(body=[cls],type_ignores=[]),'<isolated real trainer methods>','exec'),ns)
class ToyLIDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_model=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4))
        self.pnet_model=nn.Linear(4,4)
        self.offset_dec=nn.Linear(4,1)
        self.prob_dec=nn.Linear(4,1)
        self.epochs_seen=[]
    def forward(self,batch,kind,epoch):
        self.epochs_seen.append(epoch)
        x=self.pnet_model(self.resnet_model(batch))
        loss=self.offset_dec(x).square().mean()+self.prob_dec(x).square().mean()
        return True,{},dict(loss_net=loss)
t=ns['SPVNASExperiment']()
t.opt=NS(training=NS(accumulation_steps=4,nepochs=60,do_valid=False,log_interval=1000,
                    validate_before_resume=False,stop_after_epoch_file=None),
         dist=NS(ddp=False),gpu_id=0,model=NS(variant='cpu_fixture'),resume=None)
t.device=torch.device('cpu');t.lidf=ToyLIDF();t.global_step=116;t.start_epoch=58
t.best_rmse=.02;t.epoch_budget=True;t.train_sampler=None
t.train_data_loader=[torch.randn(2,3) for _ in range(10)]
t.loader_generator=torch.Generator().manual_seed(3)
t.optimizer=torch.optim.Adam([dict(name='lidf',params=list(t.lidf.parameters()),lr=.001)])
t.scheduler=torch.optim.lr_scheduler.StepLR(t.optimizer,60,gamma=.1)
ss=t.scheduler.state_dict();ss.update(last_epoch=58,_step_count=59);t.scheduler.load_state_dict(ss)
t.ckpt_dir=str(OUT/'validation/cpu_trainer_fixture');Path(t.ckpt_dir).mkdir(exist_ok=True)
t.result_dir=t.ckpt_dir;t.log_path=str(Path(t.ckpt_dir)/'steps.jsonl')
for name in ['steps.jsonl','epochs.jsonl']:
    path=Path(t.ckpt_dir)/name
    if path.exists():path.unlink()
t.training_protocol={};t.runtime_by_rank=None
def capture_fixture():t.runtime_by_rank=[capture_runtime(t.loader_generator)]
t.capture_checkpoint_runtime=capture_fixture
t.train_epoch()
saved=torch.load(Path(t.ckpt_dir)/'latest_network.pth',map_location='cpu')
assert t.completed_epochs==60 and t.global_step==120
assert t.lidf.epochs_seen==[58]*10+[59]*10
assert saved['epoch']==59 and saved['completed_epochs']==60 and saved['global_step']==120
assert saved['scheduler']['last_epoch']==60 and saved['optimizer']['param_groups'][0]['lr']==.0001
rows=[json.loads(x) for x in (Path(t.ckpt_dir)/'epochs.jsonl').read_text().splitlines()]
assert [r['discarded_microbatches'] for r in rows]==[2,2]
assert saved['runtime_by_rank'] and saved['parameter_layout']
result['actual_train_epoch_cpu_fixture']=dict(start_completed=58,end_completed=60,
    optimizer_updates=4,discarded_microbatches_each_epoch=2,
    epoch_arguments=[58,59],scheduler_last_epoch=60,passed=True)
result['syntax_valid']=True
result['cuda_initialized_after']=torch.cuda.is_initialized()
assert not result['cuda_initialized_after']
result['tested_source_files']={str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/trainers/train_spvnas.py',ROOT/'src/utils/training_resume.py',ROOT/'src/datasets/cleargrasp_fast.py',Path(__file__)]}
result['status']='passed'
(OUT/'validation/cpu_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
