"""Run two independent four-GPU experiments, then eval and timing (no baseline).

The launcher enforces successful validation and refuses existing output dirs.
This controller records all exits and never stops unrelated GPU processes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/spvnas_fusion'
sys.path.insert(0,str(Path(__file__).parent))
from launch import check_idle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus20',default='0,1,2,3')
    parser.add_argument('--gpus65',default='4,5,6,7')
    parser.add_argument('--attach', action='store_true', help='Monitor already running jobs from suite_status.json')
    a=parser.parse_args()
    if set(a.gpus20.split(',')) & set(a.gpus65.split(',')):
        parser.error('GPU groups must be disjoint')
    if len(a.gpus20.split(','))!=4 or len(a.gpus65.split(','))!=4:
        parser.error('Each experiment needs four GPUs')
    status={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'state':'running','experiments':{}}
    status_path=OUT/'suite_status.json'
    if a.attach:
        status=json.loads(status_path.read_text())
        status['baseline']='skipped at user request'
    elif status_path.exists():
        raise FileExistsError('Existing suite status: inspect before starting another suite')
    def save():
        status['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2));temporary.replace(status_path)
    def idle(gpus):
        # GPU utilization can lag after the preceding process exits.
        for attempt in range(60):
            try: return check_idle(gpus.split(','))
            except RuntimeError:
                if attempt==59: raise
                time.sleep(5)
    handles=[]
    def start(variant,gpus,port):
        idle(gpus)
        cmd=[sys.executable,str(ROOT/'scripts/spvnas/launch.py'),variant,'--gpus',gpus,'--port',str(port)]
        log=(OUT/f'suite_{variant}.log').open('x');handles.append(log)
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status['experiments'][variant]={'state':'training','pid':proc.pid,'gpus':gpus,'log':str(log.name)}
        save();print('STARTED',variant,proc.pid,gpus,flush=True)
        return proc
    try:
        if a.attach:
            class ExistingJob:
                def __init__(self, variant, pid):
                    self.variant, self.pid = variant, pid
                def poll(self):
                    proc = Path(f'/proc/{self.pid}/stat')
                    try:
                        if proc.read_text().split(') ')[1][0] != 'Z': return None
                    except FileNotFoundError:
                        pass
                    import torch
                    checkpoint=OUT/self.variant/'ckpt/stage1/latest_network.pth'
                    if not checkpoint.exists(): return 1
                    saved=torch.load(checkpoint,map_location='cpu')
                    return 0 if saved.get('global_step',0)>=10000 else 1
            jobs={v:ExistingJob(v,status['experiments'][v]['pid']) for v in
                  ['spvnas_fusion_20g','spvnas_fusion_65g']}
            save()
        else:
            jobs={'spvnas_fusion_20g':start('spvnas_fusion_20g',a.gpus20,29520),
                  'spvnas_fusion_65g':start('spvnas_fusion_65g',a.gpus65,29565)}
        while True:
            failed=False
            for variant,proc in list(jobs.items()):
                code=proc.poll()
                if code is not None:
                    status['experiments'][variant].update(state='trained' if code==0 else 'failed',exit_code=code)
                    failed |= code!=0
            save()
            if failed:
                status['state']='failed';save()
                raise RuntimeError('Experiment exited unsuccessfully; other running jobs are left intact')
            if all(p.poll()==0 for p in jobs.values()): break
            time.sleep(20)
        status['state']='evaluating';save()
        gpu=a.gpus20.split(',')[0]
        for variant in ['spvnas_fusion_20g','spvnas_fusion_65g']:
            idle(gpu)
            log=(OUT/f'suite_eval_{variant}.log').open('x');handles.append(log)
            subprocess.run([sys.executable,str(ROOT/'scripts/spvnas/launch.py'),variant,'--mode','eval','--gpus',gpu],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            idle(gpu)
            subprocess.run([sys.executable,str(ROOT/'scripts/spvnas/benchmark.py'),variant,'--gpu',gpu,
                '--checkpoint',str(OUT/variant/'ckpt/stage1/best_network.pth'),
                '--output',str(OUT/variant/'benchmark.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['experiments'][variant]['state']='evaluated_and_benchmarked';save()
        status['state']='complete';save()
        with (OUT/'REPORT.md').open('a') as report:
            report.write('\n\n## Automated suite completion\n\nCompleted at '+status['updated']+'.\n\n')
            report.write('| Variant | Parameters | Inference seconds | Peak bytes |\n|---|---:|---:|---:|\n')
            for variant in ['spvnas_fusion_20g','spvnas_fusion_65g']:
                b=json.loads((OUT/variant/'benchmark.json').read_text())
                report.write(f"| {variant} | {b['parameters']} | {b['mean_seconds']:.6f} | {b['peak_bytes']} |\n")
                metrics=OUT/variant/'evaluation/result/stage1/train_log.txt'
            report.write('\nDepth metrics are in each experiment evaluation/result/stage1/cleargrasp_test_log.txt or test_log.txt and detailed_errors/*.csv. Training budgets and checkpoint selection were identical.\n')
    except Exception as exc:
        status.update(state='failed',error=repr(exc));save();raise
    finally:
        for handle in handles: handle.close()

if __name__=='__main__':main()
