"""Official-style four-set evaluation (synthetic-val/test, real-val/test) for trained variants.

Mirrors TrainLIDF.test aggregation: per-image ClearGrasp metrics averaged over the set.
Run on an idle GPU; one process per (variant, gpu).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/spvnas'))
from entry import options

SETS = [
    ('synthetic-val', 'syn', 'known'),
    ('synthetic-test', 'syn', 'novel'),
    ('real-val', 'real', 'known'),
    ('real-test', 'real', 'novel'),
]
METRICS = ['rmse', 'rmse_log', 'abs_rel', 'sq_rel', 'log10', 'mae', 'a1', 'a2', 'a3']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=['original', 'spvnas_fusion_20g', 'spvnas_fusion_65g'])
    p.add_argument('--gpu', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--sets', nargs='*', default=[s[0] for s in SETS])
    p.add_argument('--overlay', action='append', default=[])
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()

    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,
                      CUDA_HOME='/home/quchengzou/miniconda3/envs/lidf',
                      PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
                      TORCH_EXTENSIONS_DIR=str(ROOT / 'runs/spvnas_fusion/extensions'), OMP_NUM_THREADS='2')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from utils.ipc_runtime import (check_temporary_ipc, initialize_data_worker,
                                   loader_timeout, write_progress)
    check_temporary_ipc()

    opt = options([ROOT / f'src/experiments/spvnas_fusion/{a.variant}.yaml', *a.overlay])
    opt.exp_type = 'test'
    opt.dist.ddp = False
    opt.gpu_id = 0
    opt.grid.valid_sample_num = -1
    opt.grid.miss_sample_num = -1
    checkpoint = a.checkpoint or str(ROOT / f'runs/spvnas_fusion/{a.variant}/ckpt/stage1/best_network.pth')
    out_dir = a.output_dir.resolve() if a.output_dir else Path(opt.base_log_dir, 'evaluation4')
    opt.base_log_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.set_device(0)
    from utils.runtime_settings import configure_backends
    runtime_settings = configure_backends(opt)

    os.chdir(ROOT / 'src')
    from models.pipeline import LIDF
    from datasets import cleargrasp_synthetic_dataset as syn_data
    from datasets import cleargrasp_dataset as real_data

    model = LIDF(opt, torch.device('cuda:0'))
    state = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    print('Loaded {} (variant {})'.format(checkpoint, state.get('variant')), flush=True)

    summary = {}
    for name, kind, obj in SETS:
        if name not in a.sets:
            continue
        params = opt.dataset.dict.copy()
        params['use_data_augmentation'] = False
        if kind == 'syn':
            dataset = syn_data.get_dataset(opt.dataset.cleargrasp_root_dir, params, 'test', obj)
        else:
            dataset = real_data.get_dataset(opt.dataset.cleargrasp_root_dir, params, 'test', obj)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                            worker_init_fn=initialize_data_worker,
                            timeout=loader_timeout(2, getattr(opt.training, 'loader_timeout_seconds', 120)))
        write_progress(out_dir, 'evaluating', dataset=name, images_done=0)
        acc = {m: [] for m in METRICS}
        rows, skipped = [], 0
        with torch.no_grad():
            for iteration, batch in enumerate(loader):
                success, data_dict, losses = model(batch, 'test', 0)
                if not success:
                    skipped += 1
                    continue
                row = {'image': iteration}
                for m in METRICS:
                    value = float(losses[m])
                    if not np.isfinite(value):
                        raise FloatingPointError(f'Nonfinite {m} in {name}, image {iteration}')
                    acc[m].append(value)
                    row[m] = value
                rows.append(row)
                if (iteration + 1) % 25 == 0:
                    write_progress(out_dir, 'evaluating', dataset=name, images_done=iteration + 1)
                if (iteration + 1) % 100 == 0:
                    print('{} {}/{}'.format(name, iteration + 1, len(loader)), flush=True)
        if not rows or skipped:
            raise RuntimeError(f'Incomplete evaluation {name}: {len(rows)} images, {skipped} skipped')
        mean = {m: float(np.mean(v)) for m, v in acc.items()}
        summary[name] = dict(images=len(rows), skipped=skipped, **mean)
        print('===> {}: rmse: {:.3f}, abs_rel: {:.3f}, mae: {:.3f}, a1: {:.2f}, a2: {:.2f}, a3: {:.2f}'.format(
            name, mean['rmse'], mean['abs_rel'], mean['mae'],
            mean['a1'] * 100, mean['a2'] * 100, mean['a3'] * 100), flush=True)
        with open(out_dir / '{}_{}.csv'.format(kind, obj), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image'] + METRICS)
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_progress(out_dir, 'finished', datasets=list(summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
