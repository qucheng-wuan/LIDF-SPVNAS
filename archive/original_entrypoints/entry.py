"""Run via torchrun for train; plain python for evaluation. Paths are absolute."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from opt import Params


def options(paths):
    opt = Params(str(ROOT / 'src/experiments/implicit_depth/default_config.yaml'))
    for path in paths:
        opt.update(str(path))
    return opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args = parser.parse_args()
    from utils.ipc_runtime import check_temporary_ipc
    check_temporary_ipc()
    opt = options(args.configs)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if opt.exp_type == 'train':
        world = int(os.environ.get('WORLD_SIZE', 1))
        if world != 4:
            raise ValueError('Controlled training requires torchrun with exactly four GPUs')
        if opt.training.micro_batch * world * opt.training.accumulation_steps != opt.training.batch_size:
            raise ValueError('Effective batch size mismatch')
        opt.dist.ddp = True
        opt.dist.ngpus_per_node = world
        opt.dist.dist_url = 'env://'
    else:
        opt.dist.ddp = False
    # Legacy CUDA extensions resolve paths relative to src.
    os.chdir(ROOT / 'src')
    from trainers.train_spvnas import createAndRunTrainer
    createAndRunTrainer(rank, opt)

if __name__ == '__main__':
    main()
