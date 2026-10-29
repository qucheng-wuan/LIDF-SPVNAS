"""Epoch-boundary state management; no checks added to the microbatch path."""
import hashlib
import json
import random

import numpy as np
import torch


def capture_runtime(loader_generator, device=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device) if device is not None else None,
                loader_generator=loader_generator.get_state())


def restore_runtime(state, loader_generator, device=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    loader_generator.set_state(state['loader_generator'].cpu())
    if device is not None:
        if state['torch_cuda'] is None:
            raise ValueError('Missing CUDA RNG for CUDA resume')
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def legacy_runtime(seed, rank, completed_epochs, loader_generator, device=None):
    """Only the dedicated loader stream is reconstructible in this legacy code.

    The legacy loader uses DistributedSampler and persistent_workers=False, and
    consumes one base_seed per completed train iterator. Main-process sampling
    RNG history is unavailable: use an explicit reproducible continuation seed.
    """
    loader_generator.manual_seed(seed + rank)
    for _ in range(completed_epochs):
        torch.empty((), dtype=torch.int64).random_(generator=loader_generator)
    continuation_seed = (seed + rank + 1000003 * completed_epochs) % (2**32)
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    if device is not None:
        torch.cuda.manual_seed_all(continuation_seed)
    return dict(mode='legacy_rng_incomplete', seed=continuation_seed,
                loader='reconstructed: one iterator/base_seed draw per completed epoch',
                missing=['python', 'numpy', 'torch_cpu', 'torch_cuda'],
                exact_uninterrupted_replay=False)


def parameter_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(name=g['name'], parameters=[
        dict(name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype))
        for p in g['params']]) for g in optimizer.param_groups]


def validate_checkpoint(checkpoint, model, optimizer, variant, target_epochs):
    required = {'model', 'optimizer', 'scheduler', 'global_step', 'epoch',
                'completed_epochs', 'best_rmse', 'variant', 'budget_unit'}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError('Incomplete training checkpoint: ' + str(sorted(missing)))
    if checkpoint['variant'] != variant or checkpoint['budget_unit'] != 'epochs':
        raise ValueError('Variant/budget mismatch')
    completed = checkpoint['completed_epochs']
    if not isinstance(completed, int) or not 0 < completed <= target_epochs:
        raise ValueError('Invalid completed_epochs')
    if checkpoint['epoch'] + 1 != completed:
        raise ValueError('Only complete epoch-boundary checkpoints can resume')
    if checkpoint['scheduler']['last_epoch'] != completed:
        raise ValueError('Scheduler/epoch mismatch')
    if checkpoint['global_step'] < 1 or not np.isfinite(checkpoint['best_rmse']):
        raise ValueError('Invalid training progress/best metric')
    old_groups = checkpoint['optimizer']['param_groups']
    if len(old_groups) != len(optimizer.param_groups):
        raise ValueError('Optimizer group count changed')
    for old, new in zip(old_groups, optimizer.param_groups):
        if old.get('name') != new['name'] or len(old['params']) != len(new['params']):
            raise ValueError('Optimizer parameter group order/count changed')
        for key in ('initial_lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
                    'foreach', 'capturable', 'differentiable', 'fused'):
            if old.get(key) != new.get(key):
                raise ValueError('Optimizer setting changed: ' + key)
        for pid, param in zip(old['params'], new['params']):
            state = checkpoint['optimizer']['state'].get(pid)
            if state is None or not {'step','exp_avg','exp_avg_sq'} <= state.keys():
                raise ValueError('Missing Adam state')
            for key in ('exp_avg', 'exp_avg_sq'):
                if tuple(state[key].shape) != tuple(param.shape):
                    raise ValueError('Adam state shape mismatch')
                if not torch.isfinite(state[key]).all():
                    raise ValueError('Nonfinite Adam state')
            # Current experiment uses every parameter each optimizer update.
            if int(state['step']) != checkpoint['global_step']:
                raise ValueError('Adam step/global_step mismatch')
    layout = parameter_layout(model, optimizer)
    if 'parameter_layout' in checkpoint and checkpoint['parameter_layout'] != layout:
        raise ValueError('Named optimizer parameter layout changed')
    for value in checkpoint['model'].values():
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite model checkpoint')
    return layout


def dataset_identity(dataset):
    digest = hashlib.sha256()
    for name in ('image_paths', 'mask_paths', 'depth_paths', 'json_paths'):
        for path in getattr(dataset, name):
            digest.update(str(path).encode())
            digest.update(b'\0')
    return dict(images=len(dataset), ordered_paths_sha256=digest.hexdigest())


def training_protocol(opt, dataset):
    flat = opt.get_all_keys({}, opt.dict)
    keys = [
        'seed', 'mask_type', 'training.micro_batch', 'training.accumulation_steps',
        'training.batch_size', 'training.num_workers', 'training.valid_batch_size',
        'training.nepochs', 'training.nepoch_decay', 'training.decay_gamma',
        'training.lr', 'training.lidf_lr_factor', 'training.backbone_lr_factor',
        'training.budget_unit', 'training.deterministic', 'training.allow_tf32',
        'training.do_valid', 'dist.ngpus_per_node', 'spvnas.allow_tf32',
        'spvnas.pretrained', 'spvnas.freeze_existing', 'spvnas.voxel_size',
        'spvnas.coordinate_scale', 'spvnas.spatial_shape']
    prefixes = ('model.', 'loss.', 'grid.', 'dataset.')
    protocol = {k: v for k,v in flat.items() if k in keys or k.startswith(prefixes)}
    protocol['dataset_identity'] = dataset_identity(dataset)
    # Fail early for accidental Params/non-serializable values.
    json.dumps(protocol, sort_keys=True)
    return protocol
