"""Reviewed epoch-boundary handoff and durable 20G-only continuation.

Default --plan is read-only for production. --execute-cutover must only be used
after the user approves the concrete handoff.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent
PARENT=OUT.parent
OLD_ROOT=Path('/home/quchengzou/implicit_depth')
OLD_RUN=PARENT/'spvnas_fusion_20g'
RUN=OUT/'training'
sys.path.insert(0,str(ROOT/'scripts/spvnas'))
from continuation_manifest import manifest,verify
from launch import check_idle

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)

def rows(path):
    data=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:data.append(json.loads(line))
            except json.JSONDecodeError:pass
    return data

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def identity(pid):
    p=psutil.Process(pid)
    return dict(pid=pid,created=p.create_time(),cmdline=p.cmdline())

def process(info):
    try:
        p=psutil.Process(info['pid'])
        return p if p.create_time()==info['created'] and p.status()!=psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess:return None

def signal_one(info,sig):
    p=process(info)
    if p:p.send_signal(sig)

def discover():
    selection=json.loads((PARENT/'selection_change.json').read_text())
    launcher=process(selection['launcher'])
    if launcher is None:raise RuntimeError('Original training launcher is no longer active')
    torchruns=[p for p in launcher.children() if 'torch.distributed.run' in ' '.join(p.cmdline())]
    if len(torchruns)!=1:raise RuntimeError('Expected exactly one original torchrun')
    watches=[p for p in psutil.process_iter(['cmdline']) if any(str(OLD_ROOT/'runs/spvnas_fusion/overhead_audit_20261013/watch_20g_only.py')==arg for arg in (p.info['cmdline'] or []))]
    if len(watches)!=1:raise RuntimeError('Expected exactly one legacy watcher')
    for info in selection['controllers']:
        if process(info) is None:raise RuntimeError('Legacy controller identity changed')
    return dict(launcher=identity(launcher.pid),torchrun=identity(torchruns[0].pid),
                controllers=[identity(i['pid']) for i in selection['controllers']],
                watcher=identity(watches[0].pid))

def check_parent_source():
    expected=json.loads((OUT/'provenance/parent_source_manifest.json').read_text())
    changed=[rel for rel,digest in expected['files'].items()
             if not (OLD_ROOT/rel).exists() or sha(OLD_ROOT/rel)!=digest]
    if changed:raise RuntimeError('Parent source changed: '+str(changed))

def check_cpu_evidence():
    evidence=json.loads((OUT/'validation/cpu_checks.json').read_text())
    if evidence.get('status')!='passed' or evidence.get('cuda_initialized_after'):
        raise RuntimeError('CPU validation evidence missing')
    for rel,digest in evidence['tested_source_files'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('CPU-tested source changed: '+rel)

def prepare_plan():
    check_parent_source();check_cpu_evidence()
    current=rows(OLD_RUN/'result/stage1/epochs.jsonl')[-1]
    progress=rows(OLD_RUN/'result/stage1/train_log.txt')[-1]
    source=manifest()
    plan=dict(state='ready_for_review',created=time.time(),processes=discover(),
              currently_completed_epochs=current['completed_epochs'],
              next_boundary_at_preparation=current['completed_epochs']+1,
              latest_global_step=progress['step'],gpus=[4,5,6,7],
              effective_batch=32,micro_batch=2,accumulation=4,target_total_epochs=60,
              source_signature=source['signature'],
              changes=['full training-state resume','new checkpoint per-rank RNG',
                       'disable anomaly in formal Stage 1','equivalent deterministic preprocessing'],
              missing_legacy_state=['Python RNG','NumPy RNG','Torch CPU RNG','CUDA RNG'],
              legacy_loader_generator='reconstructed from epoch iterator count; CPU checked',
              cutover_window='Next-epoch updates after saved boundary may be discarded; record observed count',
              output=str(OUT),production_stopped=False)
    atomic(OUT/'provenance/continuation_source_manifest.json',source)
    atomic(OUT/'handoff_plan.json',plan)
    return plan

def idle(gpus,timeout=300):
    start=time.time()
    while True:
        try:return check_idle([str(g) for g in gpus])
        except RuntimeError:
            if time.time()-start>timeout:raise
            time.sleep(3)

def environment():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',
        CUDA_HOME='/home/quchengzou/miniconda3/envs/lidf',
        PATH=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',''),
        TORCH_EXTENSIONS_DIR=str(ROOT/'runs/spvnas_fusion/extensions'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return env

def run_checked(cmd,log,env):
    with Path(log).open('x') as handle:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)

def freeze_boundary(epoch_row):
    import torch
    backup=OUT/'parent_checkpoint'
    backup.mkdir(exist_ok=False)
    for name in ['latest_network.pth','best_network.pth']:
        # Source is replaced atomically rather than written in place.
        with (OLD_RUN/'ckpt/stage1'/name).open('rb') as src, (backup/name).open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024)
    latest=torch.load(backup/'latest_network.pth',map_location='cpu')
    best=torch.load(backup/'best_network.pth',map_location='cpu')
    if (latest['completed_epochs']!=epoch_row['completed_epochs']
            or latest['global_step']!=epoch_row['global_step']
            or latest['epoch']+1!=latest['completed_epochs']
            or latest['scheduler']['last_epoch']!=latest['completed_epochs']
            or latest['best_rmse']!=best['best_rmse']):
        raise RuntimeError('Boundary checkpoint/log/best mismatch; old training left running')
    if latest['variant']!='spvnas_fusion_20g':raise RuntimeError('Wrong variant')
    for name in ['train_log.txt','epochs.jsonl','valid_log.txt','config.txt']:
        source=OLD_RUN/'result/stage1'/name
        if source.exists():shutil.copy2(source,backup/name)
    info=dict(epoch_row=epoch_row,files={n:sha(backup/n) for n in ['latest_network.pth','best_network.pth']},
              parent=str(OLD_RUN),frozen_at=time.time())
    atomic(backup/'manifest.json',info)
    return info

def stop_old(plan):
    identities=plan['processes']
    torchrun=process(identities['torchrun'])
    if torchrun is None:raise RuntimeError('Original torchrun exited before handoff')
    descendants=[identity(p.pid) for p in torchrun.children(recursive=True)]
    # SIGTERM is directed to torchrun; never kill the shared tmux process group.
    signal_one(identities['torchrun'],signal.SIGTERM)
    watched=[identities['torchrun']]+descendants+[identities['launcher']]
    deadline=time.time()+35
    while any(process(i) for i in watched) and time.time()<deadline:time.sleep(.25)
    for info in watched:
        if process(info):signal_one(info,signal.SIGTERM)
    deadline=time.time()+10
    while any(process(i) for i in watched) and time.time()<deadline:time.sleep(.25)
    for info in watched:
        if process(info):signal_one(info,signal.SIGKILL)
    for info in identities['controllers']:
        signal_one(info,signal.SIGTERM)
        signal_one(info,signal.SIGCONT)
    deadline=time.time()+5
    while any(process(i) for i in identities['controllers']) and time.time()<deadline:time.sleep(.2)
    for info in identities['controllers']:
        if process(info):signal_one(info,signal.SIGKILL)
    if any(process(i) for i in watched+identities['controllers']+[identities['watcher']]):
        raise RuntimeError('Legacy processes still active')
    return rows(OLD_RUN/'result/stage1/train_log.txt')[-1]

def make_configs(boundary):
    extra=dict(base_log_dir=str(RUN),resume=str(OUT/'parent_checkpoint/latest_network.pth'),
        checkpoint_path=None,debug=False,
        training=dict(fast_preprocessing=True,detect_anomaly=False,allow_legacy_rng_resume=True,
            resume_protocol_path=str(OUT/'provenance/parent_training_protocol.json'),
            resume_best_path=str(OUT/'parent_checkpoint/best_network.pth'),
            validate_before_resume=True,resume_expected_rmse=boundary['epoch_row']['validation_rmse'],
            resume_validation_tolerance=1e-6,stop_after_epoch_file=str(OUT/'STOP_AFTER_EPOCH')),
        spvnas=dict(repo_dir=str(ROOT/'third_party/spvnas'),
                    assets_dir=str(ROOT/'runs/spvnas_fusion/assets')))
    path=OUT/'configs/continuation.yaml'
    path.write_text(yaml.safe_dump(extra,sort_keys=False))
    configs=[str(ROOT/'src/experiments/spvnas_fusion/spvnas_fusion_20g.yaml'),
             str(OUT/'provenance/common.yaml'),str(path)]
    atomic(OUT/'provenance/launch_configs.json',dict(paths=configs,sha256={p:sha(p) for p in configs}))
    return configs

def monitor(proc,state,expected):
    while proc.poll() is None:
        records=rows(RUN/'result/stage1/train_log.txt')
        epochs=rows(RUN/'result/stage1/epochs.jsonl')
        if epochs:state['last_completed_epoch']=epochs[-1]
        if records:
            state['last_update']=records[-1]
            window=records[-501:]
            intervals=[b['timestamp']-a['timestamp'] for a,b in zip(window,window[1:])
                       if b['step']==a['step']+1 and a['epoch']==b['epoch']]
            if len(intervals)>=200:
                import statistics
                mean=statistics.mean(intervals)
                remaining=max(0,60*1420-records[-1]['step'])
                state['throughput']=dict(updates=len(intervals),mean_seconds_per_update=mean,
                    median_seconds_per_update=statistics.median(intervals),
                    estimated_training_only_remaining_hours=remaining*mean/3600,
                    excludes='future validation/save and final four-set evaluation; workload can vary')
        state['updated']=time.time()
        atomic(OUT/'status.json',state)
        time.sleep(10)
    if proc.returncode:raise RuntimeError('Continuation exited '+str(proc.returncode))
    verify(expected)
    import torch
    checkpoint=torch.load(RUN/'ckpt/stage1/latest_network.pth',map_location='cpu')
    if checkpoint['completed_epochs']!=60:
        if (OUT/'STOP_AFTER_EPOCH').exists():
            state.update(state='stopped_at_requested_epoch',completed_epochs=checkpoint['completed_epochs'])
            atomic(OUT/'status.json',state)
            return False
        raise RuntimeError('Training exited before total 60 epochs')
    return True

def execute():
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reviewed=json.loads((OUT/'handoff_plan.json').read_text())
    expected=json.loads((OUT/'provenance/continuation_source_manifest.json').read_text())
    verify(expected);check_parent_source();check_cpu_evidence()
    if (OUT/'parent_checkpoint').exists() or RUN.exists():
        raise FileExistsError('Handoff already started; inspect status before rerunning')
    # Discover identities anew and require the same launched experiment.
    current=discover()
    if current['launcher']['created']!=reviewed['processes']['launcher']['created']:
        raise RuntimeError('Reviewed training instance changed')
    reviewed['processes']=current
    before=rows(OLD_RUN/'result/stage1/epochs.jsonl')[-1]
    target=before['completed_epochs']+1
    state=dict(state='waiting_for_epoch_boundary',controller_pid=os.getpid(),
               target_boundary=target,started=time.time(),gpus=[4,5,6,7],source_signature=expected['signature'])
    atomic(OUT/'status.json',state)
    # Prevent the obsolete original queue from launching while keeping its TTY
    # leader alive. Training children remain running.
    # Do not stop the tmux session leader: tmux resumes its foreground
    # process group when the pane leader stops, undoing the queue suspension.
    session_leader = os.getsid(current['torchrun']['pid'])
    queue_controllers = [i for i in current['controllers'] if i['pid'] != session_leader]
    if len(queue_controllers) != 1:
        raise RuntimeError('Unexpected queue controller/session hierarchy')
    for info in queue_controllers:signal_one(info,signal.SIGSTOP)
    time.sleep(.5)
    if any(process(i) is None or process(i).status()!=psutil.STATUS_STOPPED for i in queue_controllers):
        raise RuntimeError('Could not safely pause obsolete queue controller')
    state['paused_queue_controllers'] = [i['pid'] for i in queue_controllers]
    state['preserved_session_leader'] = session_leader
    atomic(OUT/'status.json',state)
    # Kill only the old watcher. It must not treat an intentional handoff as failure.
    signal_one(current['watcher'],signal.SIGTERM)
    while True:
        if process(current['torchrun']) is None:
            raise RuntimeError('Original training exited while waiting for boundary')
        epoch_rows=rows(OLD_RUN/'result/stage1/epochs.jsonl')
        if epoch_rows and epoch_rows[-1]['completed_epochs']>=target:
            boundary=freeze_boundary(epoch_rows[-1]);break
        time.sleep(.5)
    state.update(state='stopping_old_training',boundary=boundary)
    atomic(OUT/'status.json',state)
    last=stop_old(reviewed)
    state['discarded_uncheckpointed_optimizer_updates']=max(0,last['step']-boundary['epoch_row']['global_step'])
    state['discarded_uncheckpointed_note']='Includes completed logged updates; partial microbatches may also have started'
    state['old_processes_stopped']=True
    atomic(OUT/'status.json',state)
    atomic(PARENT/'continuation_handoff.json',dict(state='continued_in_child_directory',
        child=str(OUT),boundary=boundary['epoch_row'],old_controllers_terminated=True))
    configs=make_configs(boundary)
    env=environment()
    idle([4,5,6,7]);verify(expected)
    state['state']='four_rank_preflight';atomic(OUT/'status.json',state)
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29611',
         str(ROOT/'scripts/spvnas/check_continuation_ddp.py'),'--configs',*configs,
         '--output-dir',str(OUT/'validation/ddp')]
    run_checked(cmd,OUT/'validation/ddp.log',env)
    evidence=json.loads((OUT/'validation/ddp/result.json').read_text())
    if evidence['status']!='passed' or evidence['signature']!=expected['signature']:
        raise RuntimeError('Four-rank preflight/signature check failed')
    verify(expected);idle([4,5,6,7])
    RUN.mkdir(exist_ok=False)
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29612',
         str(ROOT/'scripts/spvnas/entry.py'),'--configs',*configs]
    atomic(RUN/'launch.json',dict(command=cmd,gpus=[4,5,6,7],configs=configs,
        source_signature=expected['signature'],parent_checkpoint=boundary))
    state['state']='training';atomic(OUT/'status.json',state)
    with (RUN/'console.log').open('x') as handle:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        state['launcher_pid']=proc.pid
        complete=monitor(proc,state,expected)
    if not complete:return
    state['state']='evaluating_four_sets';atomic(OUT/'status.json',state)
    best=RUN/'ckpt/stage1/best_network.pth'
    verify(expected);idle([4])
    overlays=['--overlay',configs[1],'--overlay',configs[2]]
    cmd=[sys.executable,str(ROOT/'scripts/spvnas/eval_all_sets.py'),'spvnas_fusion_20g',
         '--gpu','4','--checkpoint',str(best),*overlays,'--output-dir',str(RUN/'evaluation4')]
    run_checked(cmd,OUT/'evaluation4.log',env)
    verify(expected);idle([4])
    cmd=[sys.executable,str(ROOT/'scripts/spvnas/benchmark.py'),'spvnas_fusion_20g',
         '--gpu','4','--checkpoint',str(best),*overlays,'--output',str(RUN/'benchmark.json')]
    run_checked(cmd,OUT/'benchmark.log',env)
    state.update(state='complete',completed_epochs=60,finished=time.time(),
                 four_sets=str(RUN/'evaluation4/summary.json'),benchmark=str(RUN/'benchmark.json'))
    atomic(OUT/'status.json',state)

def main():
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan',action='store_true')
    modes.add_argument('--execute-cutover',action='store_true')
    args=p.parse_args()
    if args.plan:print(json.dumps(prepare_plan(),ensure_ascii=False,indent=2));return
    try:execute()
    except Exception as exc:
        state=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        state.update(state='failed',error=repr(exc),updated=time.time())
        atomic(OUT/'status.json',state)
        raise

if __name__=='__main__':main()
