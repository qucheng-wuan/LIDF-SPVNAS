"""Run bounded, sequential diagnostics on four idle GPUs; no production jobs."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
import time
from launch import check_idle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/spvnas_fusion/overhead_audit_20261013'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7', CUDA_HOME='/home/quchengzou/miniconda3/envs/lidf',
           TORCH_EXTENSIONS_DIR=str(ROOT/'runs/spvnas_fusion/extensions'), OMP_NUM_THREADS='2')
env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
parser = argparse.ArgumentParser()
parser.add_argument('--stages', nargs='+', default=['before_cached_repeat', 'after_cached', 'after_fast', 'after_profile', 'after_loader'])
for name in parser.parse_args().stages:
    for attempt in range(7):
        try:
            check_idle(['4', '5', '6', '7'])
            break
        except RuntimeError:
            if attempt == 6:
                raise
            time.sleep(5)
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=4', '--master_port=29630',
           str(ROOT/'scripts/spvnas/benchmark_training.py')]
    if name != 'after_loader':
        cmd += ['--preload']
    if name == 'before_cached_repeat':
        cmd += ['--source-tree', str(OUT/'before')]
    if name == 'after_profile':
        cmd += ['--profile']
    cmd += ['--configs', str(ROOT/'src/experiments/spvnas_fusion/spvnas_fusion_20g.yaml'), str(OUT/f'{name}.yaml')]
    with (OUT/f'{name}.log').open('x') as handle:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
