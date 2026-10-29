"""Portable paths and process environment; no model or optimizer changes."""
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def absolute(value):
    if not value:
        return value
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def resolve_options(opt):
    for key in ('base_log_dir', 'checkpoint_path', 'resume'):
        setattr(opt, key, absolute(getattr(opt, key, None)))
    opt.dataset.cleargrasp_root_dir = absolute(
        os.environ.get('LIDF_DATA_ROOT') or opt.dataset.cleargrasp_root_dir)
    opt.dataset.omniverse_root_dir = absolute(opt.dataset.omniverse_root_dir)
    opt.spvnas.repo_dir = absolute(opt.spvnas.repo_dir)
    opt.spvnas.assets_dir = absolute(os.environ.get('SPVNAS_ASSETS_DIR') or opt.spvnas.assets_dir)
    for key in ('resume_protocol_path', 'resume_best_path', 'stop_after_epoch_file'):
        if hasattr(opt.training, key):
            setattr(opt.training, key, absolute(getattr(opt.training, key)))
    return opt


def setup_environment():
    # A short, user-owned path avoids AF_UNIX socket path limits. Parent and
    # torchrun/DataLoader children inherit exactly the same temporary directory.
    directory = Path(os.environ.get('LIDF_EXPECTED_TMPDIR',
                                   f'/tmp/lidf-{os.getuid()}')).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_uid != os.getuid():
        raise RuntimeError('LIDF temporary directory must belong to the current user')
    for key in ('TMPDIR', 'TMP', 'TEMP', 'LIDF_EXPECTED_TMPDIR'):
        os.environ[key] = str(directory)
    tempfile.tempdir = None
    if not os.environ.get('CUDA_HOME'):
        nvcc = Path(sys.prefix) / 'bin/nvcc'
        if not nvcc.is_file():
            found = shutil.which('nvcc')
            nvcc = Path(found).resolve() if found else None
        if nvcc is not None and nvcc.is_file():
            os.environ['CUDA_HOME'] = str(nvcc.parent.parent)
    if os.environ.get('CUDA_HOME'):
        os.environ['PATH'] = str(Path(os.environ['CUDA_HOME']) / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'runs/extensions'))
    Path(os.environ['TORCH_EXTENSIONS_DIR']).mkdir(parents=True, exist_ok=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
