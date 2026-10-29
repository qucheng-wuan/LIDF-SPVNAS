"""Own one child session; monitor real progress and bound failure cleanup."""
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def tail_rows(path, limit=100000):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size-limit))
        data = handle.read().decode(errors='replace').splitlines()
    rows = []
    for line in data:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def stop_session(proc, grace=10):
    # Every child launched here uses start_new_session=True; never signal tmux.
    def members():
        import psutil
        result=[]
        for p in psutil.process_iter(['pid','status']):
            try:
                if p.info['status'] != psutil.STATUS_ZOMBIE and os.getpgid(p.pid) == proc.pid:
                    result.append(p.pid)
            except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                pass
        return result
    def send(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    if members():
        send(signal.SIGTERM)
        deadline=time.monotonic()+grace
        while members() and time.monotonic()<deadline:
            proc.poll();time.sleep(.1)
        if members():
            send(signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL);proc.wait(timeout=5)
    deadline=time.monotonic()+5
    while members() and time.monotonic()<deadline:
        time.sleep(.1)
    if members():
        raise RuntimeError('Child session still alive: '+str(members()))


def snapshot(work, started, stage):
    work = Path(work)
    rows = tail_rows(work/'progress.json')
    progress = rows[-1] if rows else dict(phase='initializing',timestamp=started)
    result=dict(stage=stage, state=progress['phase'], progress=progress,
                last_progress_timestamp=progress['timestamp'])
    steps=tail_rows(work/'result/stage1/train_log.txt')
    epochs=tail_rows(work/'result/stage1/epochs.jsonl')
    if epochs:
        result['last_completed_epoch']=epochs[-1]
    if steps:
        result['last_update']=steps[-1]
        if progress['phase']=='training':
            result['last_progress_timestamp']=max(progress['timestamp'],steps[-1]['timestamp'])
            intervals=[b['timestamp']-a['timestamp'] for a,b in zip(steps,steps[1:])
                       if b['step']==a['step']+1 and a['epoch']==b['epoch']]
            if len(intervals)>=100:
                mean=statistics.mean(intervals[-500:])
                result['throughput']=dict(mean_seconds_per_update=mean,
                    remaining_training_hours=max(0,60*1420-steps[-1]['step'])*mean/3600,
                    excludes='validation, checkpoint saves and final four-set evaluation')
    return result


def supervise(command, *, cwd, env, log, work, status, stage,
              idle_seconds=300, startup_seconds=900, deadline_seconds=None, poll_seconds=5):
    started=time.time()
    with Path(log).open('x') as handle:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:
            while proc.poll() is None:
                state=snapshot(work,started,stage)
                state.update(controller_pid=os.getpid(),launcher_pid=proc.pid,updated=time.time(),
                             started=started,gpus=[4,5,6,7],command=command)
                age=time.time()-state['last_progress_timestamp']
                limit=startup_seconds if state['state']=='initializing' else idle_seconds
                reason=None
                if age>limit:
                    reason='No real progress for %.1fs (limit %.1fs)'%(age,limit)
                if deadline_seconds and time.time()-started>deadline_seconds:
                    reason='Bounded job deadline exceeded'
                if reason:
                    state.update(state='stalled',error=reason)
                    state.pop('throughput',None)
                    atomic(status,state)
                    raise TimeoutError(reason)
                atomic(status,state)
                time.sleep(poll_seconds)
            if proc.returncode:
                raise RuntimeError('%s exited %s; inspect %s'%(stage,proc.returncode,log))
        finally:
            stop_session(proc)
    return snapshot(work,started,stage)
