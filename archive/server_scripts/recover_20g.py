"""Single recovery from audited epoch 37; then train to 60 and evaluate four sets."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent
RUN=OUT/'training'
sys.path.insert(0,str(ROOT/'src'))
from continuation_manifest import verify
from launch import check_idle
from recovery_supervisor import atomic,supervise,tail_rows


def environment():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_HOME=str(Path(sys.executable).parents[1]),
               PATH=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',''),
               TORCH_EXTENSIONS_DIR=str(ROOT/'runs/spvnas_fusion/extensions'),
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NCCL_ASYNC_ERROR_HANDLING='1',
               TMPDIR='/ssd/polyu/lidf_tmp_quchengzou',TEMP='/ssd/polyu/lidf_tmp_quchengzou',
               TMP='/ssd/polyu/lidf_tmp_quchengzou',LIDF_EXPECTED_TMPDIR='/ssd/polyu/lidf_tmp_quchengzou')
    return env


def idle(gpus):
    deadline=time.monotonic()+90
    while True:
        try:
            return check_idle([str(x) for x in gpus])
        except RuntimeError:
            if time.monotonic()>deadline:raise
            time.sleep(3)


def interrupted(signum, frame):
    raise RuntimeError('Controller received signal '+str(signum))


def main():
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=json.loads((OUT/'provenance/recovery_source_manifest.json').read_text())
    verify(expected)
    from utils.ipc_runtime import check_temporary_ipc
    check_temporary_ipc()
    audit=json.loads((OUT/'provenance/checkpoint_audit.json').read_text())
    import hashlib
    for name,digest in audit['files'].items():
        if hashlib.sha256((OUT/'parent_checkpoint'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Backup checkpoint changed: '+name)
    for name in ('cpu_checks.json','ddp_checks.json'):
        evidence=json.loads((OUT/'validation'/name).read_text())
        if evidence.get('status')!='passed' or evidence.get('signature')!=expected['signature']:
            raise RuntimeError('Recovery evidence/signature mismatch: '+name)
    configs=json.loads((OUT/'provenance/launch_configs.json').read_text())
    for path,digest in configs['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Configuration changed: '+path)
    idle([4,5,6,7]);RUN.mkdir(exist_ok=False)
    env=environment()
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_port=29614',
         str(ROOT/'scripts/spvnas/entry.py'),'--configs',*configs['paths']]
    atomic(RUN/'launch.json',dict(command=cmd,configs=configs,source_signature=expected['signature'],
                                parent=audit,gpus=[4,5,6,7],effective_batch=32))
    supervise(cmd,cwd=ROOT,env=env,log=RUN/'console.log',work=RUN,status=OUT/'status.json',stage='training')
    verify(expected)
    import torch
    checkpoint=torch.load(RUN/'ckpt/stage1/latest_network.pth',map_location='cpu')
    completed=checkpoint['completed_epochs']
    if completed!=60:
        if (OUT/'STOP_AFTER_EPOCH').exists():
            atomic(OUT/'status.json',dict(state='stopped_at_requested_epoch',completed_epochs=completed))
            return
        raise RuntimeError('Training exited before epoch 60')
    del checkpoint
    best=RUN/'ckpt/stage1/best_network.pth'
    overlays=['--overlay',configs['paths'][1],'--overlay',configs['paths'][2]]
    idle([4]);verify(expected)
    cmd=[sys.executable,str(ROOT/'scripts/spvnas/eval_all_sets.py'),'spvnas_fusion_20g',
         '--gpu','4','--checkpoint',str(best),*overlays,'--output-dir',str(RUN/'evaluation4')]
    supervise(cmd,cwd=ROOT,env=env,log=OUT/'evaluation4.log',work=RUN/'evaluation4',
              status=OUT/'status.json',stage='evaluating_four_sets',idle_seconds=600)
    summary=json.loads((RUN/'evaluation4/summary.json').read_text())
    if set(summary)!={'synthetic-val','synthetic-test','real-val','real-test'}:
        raise RuntimeError('Four-set evaluation incomplete')
    idle([4]);verify(expected)
    cmd=[sys.executable,str(ROOT/'scripts/spvnas/benchmark.py'),'spvnas_fusion_20g',
         '--gpu','4','--checkpoint',str(best),*overlays,'--output',str(RUN/'benchmark.json')]
    supervise(cmd,cwd=ROOT,env=env,log=OUT/'benchmark.log',work=RUN/'benchmark_progress',
              status=OUT/'status.json',stage='benchmark',deadline_seconds=900)
    verify(expected)
    atomic(OUT/'status.json',dict(state='complete',completed_epochs=60,finished=time.time(),
                                 four_sets=str(RUN/'evaluation4/summary.json'),
                                 benchmark=str(RUN/'benchmark.json')))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        old=tail_rows(OUT/'status.json')
        # status JSON is indented: read it as an object rather than JSONL.
        path=OUT/'status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(state='failed',error=repr(exc),updated=time.time())
        state.pop('throughput',None)
        atomic(path,state)
        raise
