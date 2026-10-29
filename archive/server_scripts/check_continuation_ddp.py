"""Four-rank bounded preflight on the same GPUs after the old job exits.

Does not launch production, and diagnostic updates never enter its checkpoint.
"""
import argparse
import copy
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/spvnas'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--configs',nargs='+',required=True)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    from entry import options
    from continuation_manifest import manifest
    source_manifest=manifest()
    import torch
    import torch.distributed as dist
    from utils.runtime_settings import configure_backends
    from utils.training_resume import capture_runtime,restore_runtime
    rank=int(os.environ['LOCAL_RANK'])
    opt=options(a.configs)
    opt.gpu_id=rank;opt.dist.ddp=True;opt.dist.ngpus_per_node=4
    opt.dist.global_gpu_id=rank;opt.dist.world_size=4
    opt.base_log_dir=str(a.output_dir/'initial')
    opt.training.validate_before_resume=False
    os.chdir(ROOT/'src')
    torch.cuda.set_device(rank)
    configure_backends(opt)
    from trainers.train_spvnas import SPVNASExperiment
    torch.autograd.set_detect_anomaly(False)
    dist.init_process_group('nccl')
    trainer=SPVNASExperiment(opt)
    trainer.train_sampler.set_epoch(trainer.start_epoch)
    iterator=iter(trainer.train_data_loader)
    batches=[next(iterator) for _ in range(opt.training.accumulation_steps)]
    del iterator
    trainer.capture_checkpoint_runtime()
    if rank==0:
        trainer.save_checkpoint(trainer.start_epoch-1)
    dist.barrier()
    origin_path=Path(trainer.ckpt_dir)/'latest_network.pth'
    origin=torch.load(origin_path,map_location='cpu')
    runtime=origin['runtime_by_rank'][rank]
    def reset(t):
        t.network.load_state_dict(origin['model'],strict=True)
        t.optimizer.load_state_dict(copy.deepcopy(origin['optimizer']))
        t.scheduler.load_state_dict(copy.deepcopy(origin['scheduler']))
        t.optimizer.zero_grad(set_to_none=True)
        restore_runtime(runtime,t.loader_generator,t.device)
    def run_update(t,anomaly):
        torch.autograd.set_detect_anomaly(anomaly)
        t.lidf.train()
        losses=[];outputs=[]
        torch.cuda.synchronize();start=time.perf_counter()
        for i,batch in enumerate(batches):
            context=t.lidf.no_sync() if i+1<len(batches) else nullcontext()
            with context:
                success,data,loss=t.lidf(batch,'train',t.start_epoch)
                if not success:raise RuntimeError('Empty geometry in preflight')
                (loss['loss_net']/len(batches)).backward()
            losses.append(loss['loss_net'].detach().cpu())
            outputs.append(data['pred_pos'].detach().cpu())
        torch.cuda.synchronize();compute_seconds=time.perf_counter()-start
        gradients={n:p.grad.detach().cpu().clone() for n,p in t.network.named_parameters() if p.grad is not None}
        if len(gradients)!=264 or not all(torch.isfinite(v).all() for v in gradients.values()):
            raise RuntimeError('Missing/nonfinite gradients')
        t.optimizer.step()
        state={k:v.detach().cpu().clone() for k,v in t.network.state_dict().items()}
        optstate=copy.deepcopy(t.optimizer.state_dict())
        def cpu_tree(v):
            if torch.is_tensor(v):return v.detach().cpu().clone()
            if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
            if isinstance(v,list):return [cpu_tree(x) for x in v]
            return v
        optstate=cpu_tree(optstate)
        t.optimizer.zero_grad(set_to_none=True)
        return dict(losses=torch.stack(losses),outputs=outputs,gradients=gradients,
                    model=state,optimizer=optstate,compute_seconds=compute_seconds)
    reset(trainer)
    on=run_update(trainer,True)
    reset(trainer)
    off=run_update(trainer,False)
    before=dict(global_step=trainer.global_step,completed_epochs=trainer.completed_epochs,
                learning_rates=[g['lr'] for g in trainer.optimizer.param_groups])
    del trainer
    gc.collect();torch.cuda.empty_cache();dist.barrier()
    opt.resume=str(origin_path)
    opt.base_log_dir=str(a.output_dir/'restored')
    resumed=SPVNASExperiment(opt)
    assert resumed.global_step==before['global_step'] and resumed.completed_epochs==before['completed_epochs']
    restored=run_update(resumed,False)

    def difference(left,right):
        numerator=denominator=maximum=0.
        if left.keys()!=right.keys():raise AssertionError('Tensor keys changed')
        for name in left:
            x=left[name].double();y=right[name].double()
            if x.shape!=y.shape:raise AssertionError('Shape changed '+name)
            d=x-y
            numerator+=float((d*d).sum());denominator+=float((x*x).sum())
            if d.numel():maximum=max(maximum,float(d.abs().max()))
        return dict(relative_l2=(numerator/max(denominator,1e-30))**.5,max_abs=maximum)
    def comparison(left,right):
        grad=difference(left['gradients'],right['gradients'])
        model=difference(left['model'],right['model'])
        out=difference({str(i):x for i,x in enumerate(left['outputs'])},
                       {str(i):x for i,x in enumerate(right['outputs'])})
        loss=float((left['losses']-right['losses']).abs().max())
        steps=[int(s['step']) for s in right['optimizer']['state'].values()]
        # CPU equality is tested separately. CUDA sparse reductions are not
        # deterministic; report discrepancies and enforce explicit finite tolerances.
        passed=(grad['relative_l2']<1e-4 and model['relative_l2']<1e-6
                and out['relative_l2']<1e-5 and loss<1e-3
                and set(steps)=={before['global_step']+1})
        return dict(passed=passed,gradient=grad,model_after_update=model,
                    prediction=out,max_loss_difference=loss,adam_steps=sorted(set(steps)))
    record=dict(rank=rank,source_progress=before,anomaly_on_off=comparison(on,off),
                full_checkpoint_restore=comparison(off,restored),
                timed_compute_seconds=dict(on=on['compute_seconds'],off=off['compute_seconds'],
                                           restored=restored['compute_seconds']),
                caveat='Four cached microbatches; diagnostic update, not end-to-end training throughput')
    records=[None]*4
    dist.all_gather_object(records,record)
    passed=all(r[k]['passed'] for r in records for k in ('anomaly_on_off','full_checkpoint_restore'))
    if rank==0:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        (a.output_dir/'result.json').write_text(json.dumps(dict(status='passed' if passed else 'failed',ranks=records,signature=source_manifest['signature'],source_checkpoint=opt.resume),indent=2))
    dist.destroy_process_group()
    if not passed:raise RuntimeError('Four-rank continuation validation failed')

if __name__=='__main__':main()
