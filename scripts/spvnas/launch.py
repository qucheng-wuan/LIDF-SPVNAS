"""Check occupancy and smoke evidence before starting a controlled experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from project import setup_environment, absolute

ROOT = Path(__file__).resolve().parents[2]


def check_idle(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    selected = []
    for row in rows.splitlines():
        fields = [x.strip() for x in row.split(',')]
        if fields[0] in gpus:
            if fields[1] in processes or int(fields[4]) > 200 or int(fields[5]) > 5:
                raise RuntimeError('Selected GPU is busy: ' + row)
            selected.append(row)
    if len(selected) != len(gpus):
        raise ValueError('Missing GPU IDs')
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=['original','spvnas_fusion_20g','spvnas_fusion_65g'])
    p.add_argument('--gpus', required=True)
    p.add_argument('--mode', choices=['train','eval'], default='train')
    p.add_argument('--port', type=int, default=29520)
    p.add_argument('--overlay', action='append', default=[])
    p.add_argument('--output-dir', help='New output directory, resolved from current directory')
    p.add_argument('--resume', help='Completed-epoch checkpoint; also supply --resume-best')
    p.add_argument('--resume-best', help='Historical best checkpoint from the same run')
    p.add_argument('--validation-dir', type=Path, default=ROOT/'runs/spvnas_fusion/validation')
    a = p.parse_args()
    setup_environment()
    gpus = a.gpus.split(',')
    if len(set(gpus)) != len(gpus) or len(gpus) != (4 if a.mode == 'train' else 1):
        p.error('Train needs four distinct GPUs; eval needs one')
    occupancy = check_idle(gpus)
    configs = [str(ROOT / f'src/experiments/spvnas_fusion/{a.variant}.yaml')]
    if a.mode == 'eval':
        configs.append(str(ROOT / f'src/experiments/spvnas_fusion/{a.variant}_eval.yaml'))
    configs += a.overlay
    from entry import options
    opt = options(configs)
    if a.output_dir:
        opt.base_log_dir = str(Path(a.output_dir).expanduser().resolve())
    if bool(a.resume) != bool(a.resume_best):
        p.error('--resume and --resume-best must be supplied together')
    if a.resume:
        opt.resume = str(Path(a.resume).expanduser().resolve())
        opt.checkpoint_path = None
        opt.training.resume_best_path = str(Path(a.resume_best).expanduser().resolve())
    run = Path(opt.base_log_dir)
    if a.mode == 'train':
        from evidence import signature, configuration_signature
        current_signature = signature()
        for variant in [a.variant]:
            suffix = '_frozen' if opt.spvnas.freeze_existing else ''
            evidence = a.validation_dir / f'{variant}{suffix}.json'
            result = json.loads(evidence.read_text()) if evidence.exists() else {}
            if result.get('status') != 'passed' or result.get('signature') != current_signature:
                raise RuntimeError(f'Required smoke/gradient/20-step/checkpoint validation missing: {evidence}')
            if result.get('configuration_signature') != configuration_signature(opt):
                raise RuntimeError('Validation configuration differs; run validate.py with the same overlays')
            if variant == a.variant and variant != 'original' and result.get('pretrained') != bool(opt.spvnas.pretrained):
                raise RuntimeError('Smoke-test pretrained setting does not match launch configuration')
            expected_runtime = dict(deterministic=getattr(opt.training, 'deterministic', True),
                cudnn_benchmark=not getattr(opt.training, 'deterministic', True),
                allow_tf32=getattr(opt.training, 'allow_tf32', False),
                sparse_allow_tf32=getattr(opt.spvnas, 'allow_tf32', False))
            if result.get('runtime_settings') != expected_runtime:
                raise RuntimeError('Smoke-test numerical settings do not match launch configuration')
    # A new directory is mandatory, preventing accidental overwrite or auto-resume.
    run.mkdir(parents=True, exist_ok=False)
    (run / 'launch.json').write_text(json.dumps(dict(gpus=gpus, occupancy=occupancy, configs=configs), indent=2))
    overrides = {'base_log_dir': str(run), 'resume': opt.resume, 'checkpoint_path': opt.checkpoint_path}
    if getattr(opt.training, 'resume_best_path', None):
        overrides['training'] = {'resume_best_path': opt.training.resume_best_path}
    import yaml
    resolved = run / 'launch_overrides.yaml'
    resolved.write_text(yaml.safe_dump(overrides))
    configs.append(str(resolved))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpus, CUDA_HOME=os.environ.get('CUDA_HOME', sys.prefix),
               PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''),
               TORCH_EXTENSIONS_DIR=os.environ['TORCH_EXTENSIONS_DIR'], OMP_NUM_THREADS='2')
    cmd = [sys.executable]
    if a.mode == 'train':
        cmd += ['-m','torch.distributed.run','--nproc_per_node=4',f'--master_port={a.port}']
    cmd += [str(ROOT/'scripts/spvnas/entry.py'),'--configs',*configs]
    with (run/'console.log').open('x') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
