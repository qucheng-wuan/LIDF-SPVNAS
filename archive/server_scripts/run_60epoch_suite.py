"""Only original + official pretrained 20G; evaluate after both 60-epoch runs.

Start this controller detached only after single-GPU and DDP smoke tests pass.
Never terminates other jobs. Failed/incomplete work is explicitly reported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('original', 'spvnas_fusion_20g')
from entry import options
from evidence import signature
from launch import check_idle


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def report(out, status):
    lines = ['LIDF Stage 1: original vs official pretrained SPVNAS 20G, 60 epochs',
             'Updated: ' + time.strftime('%Y-%m-%d %H:%M:%S %z'),
             'Suite state: ' + status['state'],
             'Code integration: smoke tests passed before production launch.',
             '65G: 未运行 (excluded by request).',
             'Protocol: ClearGrasp synthetic train, 320x240, effective batch 32; identical augmentation, losses and mask_type=all.',
             'Validation: every completed epoch, synthetic-val; select lowest validation RMSE, never test-set selection.',
             'Adam: parameter-group learning rates are recorded in each parameter_groups.json (shared LIDF rate in both variants).',
             'StepLR: epoch-based, period 60, gamma 0.1 (original schedule; no updates after first decay).',
             'Same seed and LIDF initialization; train data RNG reset after branch construction.',
             'Latest + best checkpoints retained per experiment; full epoch history in epochs.jsonl.',
             'Inference benchmark: same GPU sequentially after both training jobs, batch 1, 240x320, 10 warmups + 50 iterations.',
             'Benchmark scope: GT-free transfer + geometry + features + prediction; one fixed synthetic-val image, not dataset-wide latency.',
             'Peak memory: torch.cuda.max_memory_allocated; not total nvidia-smi device reservation.',
             'Training inputs follow original synthetic corruption protocol; GT used for supervision, not the extra feature branch.',
             'Shared implementation fixes retained for both variants: deterministic ray tie selection; sparse unmatched points excluded.', '']
    for variant in VARIANTS:
        run = out / variant
        info = status.get('experiments', {}).get(variant, {})
        lines += [variant + ': ' + info.get('state', '未运行')]
        params = run/'parameter_groups.json'
        if params.exists():
            lines.append('Parameters: ' + str(sum(g['parameters'] for g in json.loads(params.read_text()))))
        bench = run/'benchmark.json'
        lines.append('Inference time / peak memory: ' + (bench.read_text() if bench.exists() else '未运行'))
        metrics = run/'evaluation4/summary.json'
        lines.append('Accuracy: ' + (metrics.read_text() if metrics.exists() else '未运行'))
    completed = all((out/v/'evaluation4/summary.json').exists() for v in VARIANTS)
    if completed:
        base = json.loads((out/VARIANTS[0]/'evaluation4/summary.json').read_text())
        fusion = json.loads((out/VARIANTS[1]/'evaluation4/summary.json').read_text())
        lines.append('Performance comparison (20G minus baseline, lower RMSE/REL/MAE is better):')
        for split in base:
            diff = {m: fusion[split][m]-base[split][m] for m in ('rmse','abs_rel','mae','a1','a2','a3')}
            lines.append(split + ': ' + json.dumps(diff))
            lines.append('RMSE ' + ('improved' if diff['rmse'] < 0 else 'unchanged' if diff['rmse'] == 0 else 'worsened'))
        lines.append('Single-seed comparison; no statistical significance claim. Code success does not imply accuracy improvement.')
    else:
        lines.append('Performance improvement: 未运行 / cannot conclude before matched completed evaluations.')
    if 'error' in status:
        lines.append('Failure: ' + status['error'])
    temp = out/'REPORT.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(out/'REPORT.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    allowed_roots = (ROOT/'runs/spvnas_fusion', Path('/ssd/polyu/quchengzou_lidf_spvnas_60ep_20261012'))
    if not any(out.is_relative_to(p) for p in allowed_roots):
        raise ValueError('Output must be within the experiment storage directories')
    status_path = out/'suite_status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    common = out/'configs/common.yaml'
    validation = out/'validation'
    current = signature()
    for v in VARIANTS:
        result = json.loads((validation/(v+'.json')).read_text())
        if result['status'] != 'passed' or result['signature'] != current:
            raise RuntimeError('Missing matching smoke evidence for ' + v)
        if v != 'original' and result.get('pretrained') is not True:
            raise RuntimeError('20G smoke must load official pretrained weights')
        ddp = json.loads((out/'ddp_smoke'/v/'verified.json').read_text())
        if ddp.get('signature') != current or ddp.get('completed_epochs') != 2:
            raise RuntimeError('Missing current two-epoch DDP smoke for ' + v)
        opt = options([ROOT/f'src/experiments/spvnas_fusion/{v}.yaml', common, out/f'configs/{v}.yaml'])
        assert opt.training.budget_unit == 'epochs' and opt.training.nepochs == 60
        assert opt.training.max_steps is None and not opt.training.smoke_train_images and not opt.debug
        assert opt.training.micro_batch*4*opt.training.accumulation_steps == opt.training.batch_size == 32
        assert opt.spvnas.pretrained and not opt.spvnas.freeze_existing
        assert not opt.resume and not opt.checkpoint_path
    # Freeze a reviewable source snapshot; active imports still use the repository.
    snap = out/'source_snapshot'
    for pattern in ('src/**/*.py', 'src/**/*.yaml', 'scripts/spvnas/*.py', 'third_party/spvnas/**/*.py'):
        for path in ROOT.glob(pattern):
            target = snap/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    atomic_json(out/'source_manifest.json', dict(signature=current,
                commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                files={str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}))
    status = dict(state='starting', started=time.time(), experiments={}, signature=current)
    def save():
        status['updated'] = time.time()
        atomic_json(status_path, status)
        report(out, status)
    def idle(gpus):
        while True:
            try:
                return check_idle(gpus.split(','))
            except RuntimeError as exc:
                print('Waiting for GPUs:', str(exc), flush=True)
                time.sleep(30)
    jobs, handles = {}, []
    try:
        for v, gpus, port in [('spvnas_fusion_20g','4,5,6,7',29601), ('original','0,1,2,3',29600)]:
            status['state'] = 'waiting_for_gpus'
            status['waiting_for'] = dict(variant=v, gpus=gpus)
            save()
            idle(gpus)
            handle = (out/(v+'_launcher.log')).open('x')
            handles.append(handle)
            cmd = [sys.executable, str(ROOT/'scripts/spvnas/launch.py'), v, '--gpus', gpus, '--port', str(port),
                   '--overlay', str(common), '--overlay', str(out/f'configs/{v}.yaml'), '--validation-dir', str(validation)]
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            jobs[v] = proc
            status['experiments'][v] = dict(state='training', pid=proc.pid, gpus=gpus)
            save()
        status['state'] = 'training'
        status.pop('waiting_for', None)
        while True:
            for v, proc in jobs.items():
                code = proc.poll()
                if code is not None and status['experiments'][v]['state'] == 'training':
                    if code:
                        raise RuntimeError(f'{v} training exited {code}; other job left intact')
                    import torch
                    state = torch.load(out/v/'ckpt/stage1/latest_network.pth', map_location='cpu')
                    if state.get('completed_epochs') != 60:
                        raise RuntimeError(f'{v} exited without 60 completed epochs')
                    status['experiments'][v].update(state='trained', completed_epochs=60, steps=state['global_step'])
                    del state
            save()
            if all(p.poll() == 0 for p in jobs.values()):
                break
            time.sleep(30)
        if signature() != current:
            raise RuntimeError('Source changed during training; refuse silent evaluation with different code')
        status['state'] = 'evaluating'
        save()
        # Same physical GPU, sequential measurements after all training finishes.
        for v in VARIANTS:
            checkpoint = out/v/'ckpt/stage1/best_network.pth'
            for name, cmd in [
                ('evaluation', [sys.executable, str(ROOT/'scripts/spvnas/eval_all_sets.py'), v, '--gpu', '0',
                                '--checkpoint', str(checkpoint), '--overlay', str(common),
                                '--output-dir', str(out/v/'evaluation4')]),
                ('benchmark', [sys.executable, str(ROOT/'scripts/spvnas/benchmark.py'), v, '--gpu', '0',
                               '--checkpoint', str(checkpoint), '--overlay', str(common),
                               '--output', str(out/v/'benchmark.json')])]:
                idle('0')
                with (out/(v+'_'+name+'.log')).open('x') as handle:
                    subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            status['experiments'][v]['state'] = 'evaluated_and_benchmarked'
            save()
        status['state'] = 'complete'
        save()
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        save()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
