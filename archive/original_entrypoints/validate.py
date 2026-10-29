"""Real-data GPU validation (no substitute SPVNAS). Run only on an idle GPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from entry import options
from launch import check_idle
from evidence import signature


def main():
    p=argparse.ArgumentParser()
    p.add_argument('variant',choices=['original','spvnas_fusion_20g','spvnas_fusion_65g'])
    p.add_argument('--gpu',required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--overlay', action='append', default=[])
    p.add_argument('--output-dir', type=Path, default=ROOT/'runs/spvnas_fusion/validation')
    a=p.parse_args()
    if a.steps<20: p.error('At least 20 steps required')
    suffix='_frozen' if a.frozen else ''
    out=a.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    record={'signature':signature(),'status':'not_run','variant':a.variant,'gpu':a.gpu,'checks':{}}
    report=out/(a.variant+suffix+'.json')
    try:
        record['hardware']=check_idle([a.gpu])
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, CUDA_HOME='/home/quchengzou/miniconda3/envs/lidf',
            PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH',''),
            TORCH_EXTENSIONS_DIR=str(ROOT/'runs/spvnas_fusion/extensions'), OMP_NUM_THREADS='2')
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from datasets.cleargrasp_synthetic_dataset import get_dataset
        configs=[ROOT/f'src/experiments/spvnas_fusion/{a.variant}.yaml']
        configs += a.overlay
        if a.frozen: configs.append(ROOT/f'src/experiments/spvnas_fusion/freeze_{20 if "20g" in a.variant else 65}g.yaml')
        opt=options(configs)
        # Smoke tests always exercise full mapping/finite/batch-isolation checks,
        # even though production training omits redundant tensor assertions.
        opt.spvnas.runtime_checks=True
        opt.dist.ddp=False
        opt.gpu_id=0
        opt.base_log_dir=str(out/(a.variant+suffix))
        Path(opt.base_log_dir).mkdir(exist_ok=True)
        random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
        torch.cuda.set_device(0)
        from utils.runtime_settings import configure_backends
        runtime_settings = configure_backends(opt)
        os.chdir(ROOT/'src')
        from models.pipeline import LIDF
        from trainers.train_spvnas import parameter_groups,load_base
        model=LIDF(opt,torch.device('cuda:0'))
        record['pretrained'] = bool(opt.spvnas.pretrained) if model.spvnas_branch is not None else None
        record['configs'] = [str(p) for p in configs]
        record['runtime_settings'] = runtime_settings
        if a.frozen: load_base(model,torch.load(opt.checkpoint_path,map_location='cuda:0'))
        params=opt.dataset.dict.copy()
        params['use_data_augmentation']=False
        data=get_dataset(opt.dataset.cleargrasp_root_dir,params,'test','known')
        batch=next(iter(DataLoader(data,batch_size=2,shuffle=False)))
        # Disable stochastic point subsampling for invariance/checkpoint assertions.
        record['input']={'batch_size':2,'height':opt.dataset.img_height,'width':opt.dataset.img_width}
        opt.grid.valid_sample_num=-1
        opt.grid.miss_sample_num=-1
        model.eval()
        def forward(current, phase='valid'):
            ok,values,losses=model(current,phase,0)
            assert ok,'LIDF forward reported an empty ray/voxel set'
            assert values['pred_pos'].shape==(values['total_miss_sample_num'],3)
            assert torch.isfinite(values['pred_pos']).all()
            assert all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
            return values,losses
        with torch.no_grad():
            values,checked_losses=forward(batch)
            record['checks']['forward_shape_finite']='passed'
            if model.spvnas_branch is not None:
                opt.spvnas.runtime_checks = False
                try:
                    fast_values, fast_losses = forward(batch)
                    torch.testing.assert_close(values['pred_pos'], fast_values['pred_pos'],
                                               rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(checked_losses['loss_net'], fast_losses['loss_net'],
                                               rtol=1e-5, atol=1e-6)
                finally:
                    opt.spvnas.runtime_checks = True
                record['checks']['fast_vs_checked_forward'] = 'passed'
            from inference import predict
            gt_free = {k:v for k,v in batch.items() if k not in ('xyz', 'depth')}
            inferred = predict(model, gt_free)
            torch.testing.assert_close(values['pred_pos'], inferred, rtol=1e-5, atol=1e-6)
            poisoned = dict(batch, xyz=torch.full_like(batch['xyz'], float('nan')),
                            depth=torch.full_like(batch['depth'], float('nan')))
            torch.testing.assert_close(inferred, predict(model, poisoned), rtol=1e-5, atol=1e-6)
            record['checks']['inference_without_gt_and_poisoned_gt']='passed'
            flipped={k:(v.flip(0) if torch.is_tensor(v) else list(reversed(v))) for k,v in batch.items()}
            other,_=forward(flipped)
            for bid in range(2):
                first=values['pred_pos'][values['miss_bid']==bid]
                second=other['pred_pos'][other['miss_bid']==1-bid]
                torch.testing.assert_close(first,second,rtol=1e-4,atol=1e-5)
            # Compare batched output to individual examples, detecting batch leakage.
            for bid in range(2):
                single={k:v[bid:bid+1] for k,v in batch.items()}
                one,_=forward(single)
                torch.testing.assert_close(values['pred_pos'][values['miss_bid']==bid],one['pred_pos'],rtol=1e-4,atol=1e-5)
        record['checks']['forward_shape_finite_batch_isolation']='passed'
        groups=parameter_groups(model,opt)
        optimizer=torch.optim.Adam(groups)
        losses=[]; gradients={}
        # Fixed real batch, original training loss and epoch-0 warmup selection.
        # Train-mode BN/Dropout are active except for explicitly frozen modules.
        model.train()
        frozen_buffers={k:v.clone() for k,v in model.named_buffers()} if a.frozen else {}
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            _,loss_dict=forward(batch,'train')
            loss=loss_dict['loss_net']
            loss.backward()
            for name,param in model.named_parameters():
                if not param.requires_grad: assert param.grad is None
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(),name
                    if param.grad.abs().sum()>0: gradients[name]=float(param.grad.abs().sum())
            optimizer.step(); losses.append(float(loss.detach()))
        expected=['pnet_model.'] if not a.frozen else []
        if model.spvnas_branch is not None:
            expected+=['spvnas_branch.input_adapter.','spvnas_branch.fusion.']
            if not a.frozen: expected+=['spvnas_branch.backbone.']
        for prefix in expected: assert any(k.startswith(prefix) for k in gradients),prefix
        for k,v in frozen_buffers.items(): torch.testing.assert_close(v,dict(model.named_buffers())[k],rtol=0,atol=0)
        assert np.mean(losses[-5:])<np.mean(losses[:5]),'No overall loss decrease'
        record['checks']['backward_20_steps']='passed'
        record['losses']=losses
        record['gradient_examples']={prefix:next((k,v) for k,v in gradients.items() if k.startswith(prefix)) for prefix in expected}
        model.eval()
        with torch.no_grad(): before,_=forward(batch)
        checkpoint=out/(a.variant+suffix+'.pth')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoint)
        state=torch.load(checkpoint,map_location='cuda:0')
        # Perturb, then restore, proving saved state is actually used.
        with torch.no_grad(): next(model.parameters()).add_(1)
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad(): after,_=forward(batch)
        torch.testing.assert_close(before['pred_pos'],after['pred_pos'],rtol=1e-5,atol=1e-6)
        record['checks']['checkpoint_roundtrip']='passed'
        record['parameters']=sum(x.numel() for x in model.parameters())
        # Production train-mode forward/backward verifies BN and Dropout behavior.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _,training_losses=forward(batch,'train')
        training_losses['loss_net'].backward()
        if a.frozen:
            for name,param in model.named_parameters():
                if not param.requires_grad: assert param.grad is None
            for k,v in frozen_buffers.items(): torch.testing.assert_close(v,dict(model.named_buffers())[k],rtol=0,atol=0)
        record['checks']['train_mode_backward']='passed'
        model.eval()
        record['status']='passed'
    except Exception as exc:
        record.update(status='blocked_or_failed',error=repr(exc))
        raise
    finally:
        report.write_text(json.dumps(record,indent=2))

if __name__=='__main__': main()
