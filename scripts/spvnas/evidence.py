"""Content hashes for validation, source provenance and checkpoint identity."""
import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature():
    mapping = {}
    for directory in ('src', 'scripts', 'configs', 'third_party/spvnas'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.sh', '.cu', '.cpp', '.h'):
                mapping[str(path.relative_to(ROOT))] = file_sha256(path)
    assets = Path(os.environ.get('SPVNAS_ASSETS_DIR', ROOT / 'runs/spvnas_fusion/assets'))
    for path in assets.glob('SemanticKITTI_val_SPVNAS@*GMACs/*'):
        if path.is_file() and path.name in ('net.config', 'init'):
            mapping['assets/' + str(path.relative_to(assets))] = file_sha256(path)
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def configuration_signature(opt):
    # Output locations and resume bookkeeping do not change the smoke-tested
    # architecture/loss/optimizer protocol. All other effective options do.
    flat = opt.get_all_keys({}, opt.dict, '')
    ignored = {'base_log_dir', 'checkpoint_path', 'resume', 'gpu_id', 'vis_gpu',
               'spvnas.repo_dir', 'spvnas.assets_dir', 'spvnas.runtime_checks'}
    flat = {k: v for k, v in flat.items() if k not in ignored and
            not k.startswith(('dist.', 'training.resume_', 'training.validate_before_resume',
                              'training.stop_after_epoch_file'))}
    if opt.spvnas.freeze_existing and opt.checkpoint_path:
        flat['initial_checkpoint_sha256'] = file_sha256(opt.checkpoint_path)
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()
