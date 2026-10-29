"""Persistent preflight -> two smoke tests -> DDP epoch tests -> formal suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
from evidence import signature
from run_60epoch_suite import atomic_json
from launch import check_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke-gpus', help='Reuse four idle GPUs sequentially for DDP smoke tests')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    smoke_gpus = args.smoke_gpus.split(',') if args.smoke_gpus else None
    if smoke_gpus and (len(smoke_gpus) != 4 or len(set(smoke_gpus)) != 4):
        parser.error('--smoke-gpus requires four distinct GPU IDs')
    common = out/'configs/common.yaml'
    status = dict(state='smoke_tests', started=time.time())
    def save():
        status['updated'] = time.time()
        atomic_json(out/'bootstrap_status.json', status)
        print(status, flush=True)
    def parallel(commands):
        jobs, handles = [], []
        try:
            for name, command in commands:
                handle = (out/(name+'.log')).open('w')
                handles.append(handle)
                jobs.append((name, subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)))
            results = [(name, p.wait()) for name, p in jobs]
            if any(code for _, code in results):
                raise RuntimeError(str(results))
        finally:
            for handle in handles:
                handle.close()
    def wait_for_gpus(next_stage):
        status['state'] = 'waiting_for_gpus'
        status['next_stage'] = next_stage
        while True:
            try:
                check_idle(smoke_gpus or [str(i) for i in range(8)])
                status['state'] = next_stage
                status.pop('waiting_reason', None)
                save()
                return
            except RuntimeError as exc:
                status['waiting_reason'] = str(exc)
                save()
                time.sleep(30)
    try:
        wait_for_gpus('smoke_tests')
        variants = [('original','0','0,1,2,3',29700), ('spvnas_fusion_20g','4','4,5,6,7',29701)]
        if smoke_gpus:
            variants = [(v, smoke_gpus[i], ','.join(smoke_gpus), port)
                        for i, (v, _, _, port) in enumerate(variants)]
        parallel([(v+'_single_gpu_smoke', [sys.executable, str(ROOT/'scripts/spvnas/validate.py'), v,
                  '--gpu', gpu, '--overlay', str(common), '--output-dir', str(out/'validation')])
                  for v, gpu, _, _ in variants])
        wait_for_gpus('ddp_epoch_smoke')
        ddp_commands = [(v+'_ddp_smoke', [sys.executable, str(ROOT/'scripts/spvnas/launch.py'), v,
                  '--gpus', gpus, '--port', str(port), '--validation-dir', str(out/'validation'),
                  '--overlay', str(common), '--overlay', str(out/'configs/epoch_smoke.yaml'),
                  '--overlay', str(out/f'configs/smoke_{v}.yaml')]) for v, _, gpus, port in variants]
        if smoke_gpus:
            for command in ddp_commands:
                wait_for_gpus('ddp_epoch_smoke')
                parallel([command])
        else:
            parallel(ddp_commands)
        import torch
        for v, _, _, _ in variants:
            run = out/'ddp_smoke'/v
            ckpt = torch.load(run/'ckpt/stage1/latest_network.pth', map_location='cpu')
            rows = [json.loads(x) for x in (run/'result/stage1/epochs.jsonl').read_text().splitlines()]
            assert ckpt['completed_epochs'] == 2 and ckpt['epoch'] == 1
            assert ckpt['global_step'] == 4, '2 full epochs must ignore max_steps=1'
            assert ckpt['scheduler']['last_epoch'] == 2, 'scheduler must advance once per epoch'
            assert [r['completed_epochs'] for r in rows] == [1,2]
            assert all(r['validation_rmse'] is not None for r in rows)
            assert (run/'ckpt/stage1/best_network.pth').exists()
            if v != 'original':
                audit = json.loads((run/'weight_audit_rank0.json').read_text())
                assert audit['pretrained'] and audit['all_loaded_tensors_match_official_checkpoint']
            atomic_json(run/'verified.json', dict(status='passed', signature=signature(),
                        completed_epochs=2, optimizer_steps=4, validations=2, scheduler_steps=2,
                        ignores_step_cap=True, four_gpu_ddp=True))
            del ckpt
        wait_for_gpus('formal_training')
        subprocess.run([sys.executable, str(ROOT/'scripts/spvnas/run_60epoch_suite.py'),
                        '--output-dir', str(out)], cwd=ROOT, check=True)
        status['state'] = 'complete'
        save()
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
