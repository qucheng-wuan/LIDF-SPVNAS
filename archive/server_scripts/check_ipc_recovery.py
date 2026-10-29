"""Bounded CPU checks and four-rank fault injection, separate from training."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from utils.ipc_runtime import check_temporary_ipc,initialize_data_worker,loader_timeout,rank_zero_action
from continuation_manifest import manifest
from recovery_supervisor import atomic,supervise


class SlowDataset:
    def __len__(self):return 8
    def __getitem__(self,index):
        time.sleep(3)
        return index


def cpu():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    ipc=check_temporary_ipc()
    after=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    assert before[0]==after[0] and all(np.array_equal(a,b) for a,b in zip(before[1],after[1]))
    assert torch.equal(before[2],after[2])
    dataset=TensorDataset(torch.arange(32))
    def collect(init):
        gen=torch.Generator().manual_seed(2026)
        loader=DataLoader(dataset,batch_size=2,num_workers=4,shuffle=True,
                          generator=gen,worker_init_fn=init,timeout=loader_timeout(4))
        values=[torch.cat([b[0] for b in loader]) for _ in range(2)]
        return values,gen.get_state()
    old,old_state=collect(None)
    new,new_state=collect(initialize_data_worker)
    assert all(torch.equal(a,b) for a,b in zip(old,new)) and torch.equal(old_state,new_state)
    assert loader_timeout(0)==0
    zero=DataLoader(dataset,batch_size=4,num_workers=0,timeout=loader_timeout(0))
    assert torch.equal(torch.cat([b[0] for b in zero]),torch.arange(32))
    timeout_error=None
    try:
        next(iter(DataLoader(SlowDataset(),num_workers=1,timeout=.5,
                             worker_init_fn=initialize_data_worker)))
    except RuntimeError as exc:
        timeout_error=str(exc)
    assert timeout_error and 'timed out' in timeout_error
    # Same supervisor as production: an alive process is not valid progress.
    fixture=OUT/'validation/supervisor_fixture'
    fixture.mkdir(exist_ok=False)
    cmd=[sys.executable,'-c',
         'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True); time.sleep(60)']
    try:
        supervise(cmd,cwd=fixture,env=os.environ.copy(),log=fixture/'console.log',work=fixture,
                  status=fixture/'status.json',stage='fixture',startup_seconds=1,poll_seconds=.1)
        raise AssertionError('Alive-but-stalled fixture was not detected')
    except TimeoutError:
        pass
    import psutil
    child=int((fixture/'console.log').read_text().strip())
    assert not psutil.pid_exists(child) or psutil.Process(child).status()==psutil.STATUS_ZOMBIE
    assert json.loads((fixture/'status.json').read_text())['state']=='stalled'
    # Startup must refuse an unexpected cached fallback even if it is writable.
    expected=os.environ['LIDF_EXPECTED_TMPDIR']
    os.environ['LIDF_EXPECTED_TMPDIR']=expected+'/wrong'
    try:
        check_temporary_ipc()
        raise AssertionError('Bad temp environment accepted')
    except RuntimeError:
        pass
    finally:
        os.environ['LIDF_EXPECTED_TMPDIR']=expected
    assert not torch.cuda.is_initialized()
    record=dict(status='passed',signature=manifest()['signature'],ipc=ipc,
        data_order_and_loader_rng_unchanged=True,main_rng_unchanged=True,
        cycles=2,workers=4,zero_workers_passed=True,timeout_error=timeout_error,
        alive_but_stalled_detected=True,child_session_cleaned=True,cuda_initialized=False)
    atomic(OUT/'validation/cpu_checks.json',record)
    print(json.dumps(record),flush=True)


def ddp(mode):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from datetime import timedelta
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    check_temporary_ipc()
    dist.init_process_group('nccl',timeout=timedelta(seconds=40))
    value=torch.tensor(float(rank),device='cuda')
    dist.all_reduce(value)
    assert value.item()==6
    started=time.monotonic()
    out=OUT/'validation'/('ddp_'+mode)
    out.mkdir(exist_ok=True)
    if mode=='hang':
        atomic(out/('rank%d.json'%rank),dict(rank=rank,state='started'))
        # An unresponsive rank must not leave the remaining GPUs occupied.
        if rank==0:
            time.sleep(120)
        else:
            dist.barrier()
        raise AssertionError('Supervisor failed to terminate injected hang')
    error=None
    def validation():
        next(iter(DataLoader(SlowDataset(),num_workers=1,timeout=.5,
                             worker_init_fn=initialize_data_worker)))
    try:
        rank_zero_action(validation,'injected validation timeout',True)
    except RuntimeError as exc:
        error=str(exc)
    atomic(out/('rank%d.json'%rank),dict(rank=rank,error=error,seconds=time.monotonic()-started))
    dist.destroy_process_group()
    if not error or 'DataLoader timed out' not in error:
        raise AssertionError('Expected coordinated timeout was not received')
    # Production must exit nonzero for a failed validation, even after propagation.
    raise RuntimeError(error)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['cpu','timeout','hang'])
    args=parser.parse_args()
    cpu() if args.mode=='cpu' else ddp(args.mode)
