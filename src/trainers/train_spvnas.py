"""Controlled Stage-1 experiments. Legacy train_lidf and Stage 2 stay intact."""
import json
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from trainers.train_lidf import TrainLIDF
from datasets import cleargrasp_synthetic_dataset as synthetic
from datasets import cleargrasp_dataset as cleargrasp
from models.pipeline import LIDF
from utils.training_resume import (capture_runtime, restore_runtime, legacy_runtime,
    validate_checkpoint, parameter_layout, training_protocol)
from utils.ipc_runtime import (check_temporary_ipc, initialize_data_worker,
    loader_timeout, rank_zero_action, write_progress)

BASE_MODULES = ('resnet_model', 'pnet_model', 'offset_dec', 'prob_dec')


def load_base(model, checkpoint):
    """Strict loading: unlike the legacy restore(), never tolerate shape mismatch."""
    for name in BASE_MODULES:
        getattr(model, name).load_state_dict(checkpoint[name], strict=True)


def parameter_groups(model, opt):
    base = [p for name in BASE_MODULES for p in getattr(model, name).parameters()]
    groups = []
    branch = model.spvnas_branch
    if opt.spvnas.freeze_existing:
        if branch is None:
            raise ValueError('Frozen baseline has no trainable adapters')
        for p in base + list(branch.backbone.parameters()):
            p.requires_grad_(False)
        model.freeze_existing = True
    else:
        groups.append(dict(name='lidf', params=base, lr=opt.training.lr *
                          getattr(opt.training, 'lidf_lr_factor', opt.training.backbone_lr_factor)))
        if branch is not None:
            groups.append(dict(name='spvnas', params=list(branch.backbone.parameters()),
                               lr=opt.training.lr * opt.training.backbone_lr_factor))
    if branch is not None:
        groups.append(dict(name='adapters', params=list(branch.input_adapter.parameters()) + list(branch.fusion.parameters()),
                           lr=opt.training.lr))
    grouped = [id(p) for g in groups for p in g['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    return groups


class SPVNASExperiment(TrainLIDF):
    def setup_model(self):
        self.lidf = LIDF(self.opt, self.device)
        self.global_step = 0
        self.start_epoch = 0
        self.best_rmse = float('inf')
        self.completed_epochs = 0
        self._resume_checkpoint = None
        self.runtime_by_rank = None
        resume = getattr(self.opt, 'resume', None) if self.opt.exp_type == 'train' else None
        if resume and self.opt.checkpoint_path:
            raise ValueError('Use resume for full continuation; do not also set checkpoint_path')
        if self.opt.spvnas.freeze_existing and not self.opt.checkpoint_path:
            raise ValueError('Frozen experiment requires a verified Stage 1 checkpoint_path')
        if resume or self.opt.checkpoint_path:
            checkpoint = torch.load(resume or self.opt.checkpoint_path, map_location='cpu')
            if resume:
                self._resume_checkpoint = checkpoint
            if 'model' in checkpoint:
                self.lidf.load_state_dict(checkpoint['model'], strict=True)
            else:
                load_base(self.lidf, checkpoint)
                if self.opt.exp_type == 'test' and self.lidf.spvnas_branch is not None:
                    raise ValueError('Fusion evaluation requires a trained fusion checkpoint')
        elif self.opt.exp_type == 'test':
            raise ValueError('Evaluation requires checkpoint_path')
        groups = parameter_groups(self.lidf, self.opt)
        summary = [dict(name=g['name'], lr=g['lr'], parameters=sum(p.numel() for p in g['params'])) for g in groups]
        print('Parameter groups:', json.dumps(summary), flush=True)
        if self.opt.gpu_id == 0:
            Path(self.opt.base_log_dir, 'parameter_groups.json').write_text(json.dumps(summary, indent=2))
        self.optimizer = torch.optim.Adam(groups)
        self.epoch_budget = getattr(self.opt.training, 'budget_unit', 'steps') == 'epochs'
        decay = self.opt.training.nepoch_decay if self.epoch_budget else self.opt.training.decay_steps
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, decay,
                                                          gamma=self.opt.training.decay_gamma)
        if self._resume_checkpoint is not None:
            if not self.epoch_budget:
                raise ValueError('Full continuation is supported only at epoch boundaries')
            checkpoint = self._resume_checkpoint
            validate_checkpoint(checkpoint, self.lidf, self.optimizer,
                                self.opt.model.variant, self.opt.training.nepochs)
            if checkpoint['scheduler']['step_size'] != decay or checkpoint['scheduler']['gamma'] != self.opt.training.decay_gamma:
                raise ValueError('Scheduler definition changed')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.global_step = checkpoint['global_step']
            self.start_epoch = self.completed_epochs = checkpoint['completed_epochs']
            self.best_rmse = checkpoint['best_rmse']
        if self.opt.dist.ddp:
            # Conversion before constructing optimizer is unnecessary: conversion
            # preserves Parameter objects; frozen BN remains eval via LIDF.train.
            if self.lidf.spvnas_branch is None:
                self.lidf = nn.SyncBatchNorm.convert_sync_batchnorm(self.lidf)
            else:
                from models.spvnas_fusion import convert_sparse_sync_batchnorm
                self.lidf = convert_sparse_sync_batchnorm(self.lidf)
            self.lidf = nn.parallel.DistributedDataParallel(self.lidf, device_ids=[self.opt.gpu_id],
                                                             find_unused_parameters=True)

        # Branch construction/determinization must not change data sampling RNG.
        random.seed(self.opt.seed + self.opt.gpu_id)
        np.random.seed(self.opt.seed + self.opt.gpu_id)
        torch.manual_seed(self.opt.seed + self.opt.gpu_id)
        torch.cuda.manual_seed_all(self.opt.seed + self.opt.gpu_id)

    def validate(self, *args, **kwargs):
        previous = self.opt.grid.valid_sample_num
        self.opt.grid.valid_sample_num = -1
        try:
            return super().validate(*args, **kwargs)
        finally:
            self.opt.grid.valid_sample_num = previous

    def progress(self, phase, **fields):
        if self.opt.gpu_id == 0:
            self._phase = phase
            write_progress(self.opt.base_log_dir, phase, global_step=self.global_step,
                           completed_epochs=self.completed_epochs, **fields)

    def run_iteration(self, epoch, iteration, iter_len, exp_type, vis_iter, batch):
        result = super().run_iteration(epoch, iteration, iter_len, exp_type, vis_iter, batch)
        # CPU counters only, after successful work; no new CUDA synchronization.
        if exp_type != 'train' and (iteration % 25 == 0 or iteration + 1 == iter_len):
            self.progress(getattr(self, '_phase', 'validating'), epoch=epoch + 1,
                          images_done=iteration + 1, batches_total=iter_len)
        return result

    def test(self, *args, **kwargs):
        previous = self.opt.grid.valid_sample_num
        self.opt.grid.valid_sample_num = -1
        try:
            return super().test(*args, **kwargs)
        finally:
            self.opt.grid.valid_sample_num = previous

    @property
    def network(self):
        return self.lidf.module if self.opt.dist.ddp else self.lidf

    def setup_data(self):
        check_temporary_ipc()
        timeout = loader_timeout(self.opt.training.num_workers,
                                 getattr(self.opt.training, 'loader_timeout_seconds', 120))
        self.synthetic_loader = synthetic.get_dataset
        if getattr(self.opt.training, 'fast_preprocessing', False):
            from datasets.cleargrasp_fast import get_dataset
            self.synthetic_loader = get_dataset
        params = self.opt.dataset.dict.copy()
        if self.opt.dataset.type != 'cleargrasp':
            raise ValueError('Controlled experiment uses only ClearGrasp synthetic data')
        if self.opt.exp_type == 'train':
            dataset = self.synthetic_loader(self.opt.dataset.cleargrasp_root_dir, params, 'train')
            subset = getattr(self.opt.training, 'smoke_train_images', 0)
            if subset:
                if not self.opt.debug:
                    raise ValueError('Dataset truncation is only allowed in explicit debug smoke runs')
                dataset = torch.utils.data.Subset(dataset, range(min(subset, len(dataset))))
            self.train_sampler = DistributedSampler(dataset, seed=self.opt.seed) if self.opt.dist.ddp else None
            self.loader_generator = torch.Generator().manual_seed(self.opt.seed + self.opt.gpu_id)
            self.training_protocol = training_protocol(self.opt, dataset)
            self.train_data_loader = DataLoader(dataset, batch_size=self.opt.training.micro_batch,
                sampler=self.train_sampler, shuffle=self.train_sampler is None,
                num_workers=self.opt.training.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=initialize_data_worker,
                generator=self.loader_generator, timeout=timeout)
        if self.opt.gpu_id == 0:
            dataset = self.synthetic_loader(self.opt.dataset.cleargrasp_root_dir, params, 'test', 'known')
            self.syn_known_test_data_loader = DataLoader(dataset, batch_size=self.opt.training.valid_batch_size,
                shuffle=False, num_workers=self.opt.training.num_workers, worker_init_fn=initialize_data_worker, timeout=timeout)
            if self.opt.exp_type == 'test':
                dataset = self.synthetic_loader(self.opt.dataset.cleargrasp_root_dir, params, 'test', 'novel')
                self.syn_novel_test_data_loader = DataLoader(dataset, batch_size=self.opt.training.valid_batch_size,
                    shuffle=False, num_workers=self.opt.training.num_workers, worker_init_fn=initialize_data_worker, timeout=timeout)
                dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, 'test', 'known')
                self.real_known_test_data_loader = DataLoader(dataset, batch_size=self.opt.training.valid_batch_size,
                    shuffle=False, num_workers=self.opt.training.num_workers, worker_init_fn=initialize_data_worker, timeout=timeout)
                dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, 'test', 'novel')
                self.real_novel_test_data_loader = DataLoader(dataset, batch_size=self.opt.training.valid_batch_size,
                    shuffle=False, num_workers=self.opt.training.num_workers, worker_init_fn=initialize_data_worker, timeout=timeout)

        if self._resume_checkpoint is not None:
            checkpoint = self._resume_checkpoint
            expected = checkpoint.get('training_protocol')
            if expected is None:
                protocol_path = getattr(self.opt.training, 'resume_protocol_path', None)
                if not protocol_path:
                    raise ValueError('Legacy resume requires reviewed parent training protocol')
                expected = json.loads(Path(protocol_path).read_text())
            if expected != self.training_protocol:
                changed = sorted(k for k in set(expected) | set(self.training_protocol)
                                 if expected.get(k) != self.training_protocol.get(k))
                raise ValueError('Training protocol changed: ' + str(changed))
            states = checkpoint.get('runtime_by_rank')
            if states:
                if len(states) != self.opt.dist.ngpus_per_node:
                    raise ValueError('RNG world size changed')
                restore_runtime(states[self.opt.gpu_id], self.loader_generator, self.device)
                rng_info = dict(mode='restored_all_rank_states')
            else:
                if not getattr(self.opt.training, 'allow_legacy_rng_resume', False):
                    raise ValueError('Legacy checkpoint lacks RNG; explicit reviewed allowance required')
                rng_info = legacy_runtime(self.opt.seed, self.opt.gpu_id, self.completed_epochs,
                                          self.loader_generator, self.device)
            self.resume_info = dict(checkpoint=str(self.opt.resume), completed_epochs=self.completed_epochs,
                global_step=self.global_step, best_rmse=self.best_rmse,
                learning_rates=[g['lr'] for g in self.optimizer.param_groups], rng=rng_info,
                parameter_layout=parameter_layout(self.network, self.optimizer))
            Path(self.opt.base_log_dir, 'resume_rank%d.json' % self.opt.gpu_id).write_text(
                json.dumps(self.resume_info, indent=2))
            if self.opt.gpu_id == 0:
                best_path = getattr(self.opt.training, 'resume_best_path', None)
                if not best_path:
                    raise ValueError('Continuation must retain the historical best checkpoint')
                previous_best = torch.load(best_path, map_location='cpu')
                if previous_best['variant'] != self.opt.model.variant or previous_best['best_rmse'] != self.best_rmse:
                    raise ValueError('Historical best checkpoint does not match latest best_rmse')
                shutil.copy2(best_path, Path(self.ckpt_dir, 'best_network.pth'))
                del previous_best
            # Do not retain a full CPU copy of model/Adam after startup.
            self._resume_checkpoint = None

    def capture_checkpoint_runtime(self):
        state = capture_runtime(self.loader_generator, self.device)
        if self.opt.dist.ddp:
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, state)
        else:
            states = [state]
        self.runtime_by_rank = states

    def validate_before_resume(self):
        if not getattr(self.opt.training, 'validate_before_resume', False):
            return
        state = capture_runtime(self.loader_generator, self.device)
        self.progress('resume_validation', epoch=self.start_epoch)
        error = [None]
        if self.opt.gpu_id == 0:
            wrapped = self.lidf
            try:
                self.lidf = self.network
                self.validate(self.syn_known_test_data_loader, 'valid', self.start_epoch - 1, 'resume-check')
                value = self.rmse.avg
                expected = self.opt.training.resume_expected_rmse
                delta = abs(value - expected)
                record = dict(rmse=value, expected_rmse=expected, absolute_difference=delta,
                              completed_epochs=self.completed_epochs, global_step=self.global_step)
                Path(self.opt.base_log_dir, 'resume_validation.json').write_text(json.dumps(record, indent=2))
                if not np.isfinite(value) or self.rmse.count == 0 or delta > self.opt.training.resume_validation_tolerance:
                    raise ValueError('Resume validation differs from the source checkpoint: ' + str(record))
            except Exception as exc:
                error[0] = repr(exc)
            finally:
                self.lidf = wrapped
        if self.opt.dist.ddp:
            dist.broadcast_object_list(error, src=0)
        restore_runtime(state, self.loader_generator, self.device)
        if error[0]:
            raise RuntimeError(error[0])

    def save_checkpoint(self, epoch, best=False):
        checkpoint = dict(model=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(), global_step=self.global_step, epoch=epoch,
            best_rmse=self.best_rmse, variant=self.opt.model.variant)
        checkpoint.update(format_version=2, runtime_by_rank=self.runtime_by_rank,
                          parameter_layout=parameter_layout(self.network, self.optimizer),
                          training_protocol=getattr(self, 'training_protocol', None),
                          parent_resume=getattr(self.opt, 'resume', None),
                          completed_epochs=getattr(self, 'completed_epochs', 0),
                          budget_unit='epochs' if self.epoch_budget else 'steps')
        for name in BASE_MODULES:
            checkpoint[name] = getattr(self.network, name).state_dict()
        for filename in (['latest_network.pth', 'best_network.pth'] if best else ['latest_network.pth']):
            path = Path(self.ckpt_dir, filename)
            temporary = path.with_suffix('.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(path)

    def train_epoch(self):
        accum = self.opt.training.accumulation_steps
        epoch = self.start_epoch
        self.completed_epochs = self.start_epoch
        self.validate_before_resume()
        def unfinished():
            return epoch < self.opt.training.nepochs if self.epoch_budget else self.global_step < self.opt.training.max_steps
        while unfinished():
            self.progress('training', epoch=epoch + 1)
            epoch_started = time.time()
            step_at_start = self.global_step
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.lidf.train()
            self.optimizer.zero_grad(set_to_none=True)
            pending, loss_sum = 0, torch.zeros((), device=self.device)
            for batch in self.train_data_loader:
                # Both forward and backward must be inside no_sync. Geometry
                # failure collectives and SyncBatchNorm remain active on all ranks.
                defer_sync = self.opt.dist.ddp and pending + 1 < accum
                context = self.lidf.no_sync() if defer_sync else nullcontext()
                with context:
                    success, _, losses = self.lidf(batch, 'train', epoch)
                    if not success:
                        self.optimizer.zero_grad(set_to_none=True)
                        pending = 0
                        loss_sum.zero_()
                        continue
                    loss = losses['loss_net']
                    (loss / accum).backward()
                pending += 1
                loss_sum.add_(loss.detach() / accum)
                if pending < accum:
                    continue
                # One reduction/host read per update, also checking every rank's
                # accumulated loss BEFORE any optimizer mutation. NaN/Inf from
                # any microbatch propagates into this sum and aborts the run.
                loss_value = loss_sum.clone()
                if self.opt.dist.ddp:
                    dist.all_reduce(loss_value)
                    loss_value /= dist.get_world_size()
                # CUDA-stream assertion: no GPU scalar read for Python control flow.
                torch._assert_async(torch.isfinite(loss_value), 'Nonfinite accumulated training loss')
                self.optimizer.step()
                if not self.epoch_budget:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.opt.gpu_id == 0:
                    row = dict(step=self.global_step, epoch=epoch, loss=loss_value.item(),
                               timestamp=time.time(), learning_rates=self.scheduler.get_last_lr())
                    with open(self.log_path, 'a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    if self.global_step % self.opt.training.log_interval == 0:
                        print(row, flush=True)
                pending = 0
                loss_sum.zero_()
                if not self.epoch_budget and (self.global_step % self.opt.training.eval_interval == 0 or self.global_step == self.opt.training.max_steps):
                    if self.opt.dist.ddp:
                        dist.barrier()
                    if self.opt.gpu_id == 0:
                        # Avoid rank-0-only DDP buffer broadcasts during validation.
                        wrapped = self.lidf
                        self.lidf = self.network
                        self.validate(self.syn_known_test_data_loader, 'valid', epoch, 'synthetic-val')
                        self.lidf = wrapped
                        improved = self.rmse.avg < self.best_rmse
                        self.best_rmse = min(self.best_rmse, self.rmse.avg)
                        self.save_checkpoint(epoch, improved)
                    if self.opt.dist.ddp:
                        dist.barrier()
                    self.lidf.train()
                if not self.epoch_budget and self.global_step >= self.opt.training.max_steps:
                    break
            if self.global_step == step_at_start:
                raise RuntimeError('No optimizer updates in epoch; inspect empty/invalid input batches')
            # An incomplete accumulation is discarded identically in every variant.
            self.optimizer.zero_grad(set_to_none=True)
            if self.epoch_budget:
                self.completed_epochs = epoch + 1
                self.scheduler.step()
                if self.opt.dist.ddp:
                    dist.barrier()
                def validate_epoch():
                    improved = False
                    if self.opt.training.do_valid:
                        wrapped = self.lidf
                        self.progress('validating', epoch=epoch + 1)
                        try:
                            self.lidf = self.network
                            self.validate(self.syn_known_test_data_loader, 'valid', epoch, 'synthetic-val')
                        finally:
                            self.lidf = wrapped
                        if not np.isfinite(self.rmse.avg) or self.rmse.count == 0:
                            raise FloatingPointError('Validation produced no finite RMSE')
                        improved = self.rmse.avg < self.best_rmse
                        self.best_rmse = min(self.best_rmse, self.rmse.avg)
                    return improved
                improved = rank_zero_action(validate_epoch, 'epoch validation', self.opt.dist.ddp)
                self.progress('checkpointing', epoch=epoch + 1)
                self.capture_checkpoint_runtime()
                def save_epoch():
                    self.save_checkpoint(epoch, improved)
                    row = dict(completed_epochs=self.completed_epochs, target_epochs=self.opt.training.nepochs,
                               global_step=self.global_step, updates_in_epoch=self.global_step-step_at_start,
                               discarded_microbatches=pending, seconds=time.time()-epoch_started,
                               validation_rmse=self.rmse.avg if self.opt.training.do_valid else None,
                               best_rmse=self.best_rmse if self.opt.training.do_valid else None,
                               timestamp=time.time())
                    with open(Path(self.result_dir, 'epochs.jsonl'), 'a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    print('EPOCH_COMPLETE', json.dumps(row), flush=True)
                rank_zero_action(save_epoch, 'epoch checkpoint', self.opt.dist.ddp)
                if self.opt.dist.ddp:
                    dist.barrier()
            epoch += 1
            # Optional external request is checked only after a complete checkpoint.
            stop_file = getattr(self.opt.training, 'stop_after_epoch_file', None)
            stop = [bool(stop_file and Path(stop_file).exists()) if self.opt.gpu_id == 0 else False]
            if self.opt.dist.ddp:
                dist.broadcast_object_list(stop, src=0)
            if stop[0]:
                if self.opt.gpu_id == 0:
                    print('STOPPED_AT_EPOCH_BOUNDARY', self.completed_epochs, flush=True)
                break
        self.progress('finished')


def createAndRunTrainer(gpu_id, opt):
    opt.gpu_id = gpu_id
    check_temporary_ipc()
    if gpu_id == 0:
        write_progress(opt.base_log_dir, 'initializing')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.set_device(gpu_id)
    torch.cuda.manual_seed_all(opt.seed)
    from utils.runtime_settings import configure_backends
    configure_backends(opt)
    # implicit_net enables anomaly detection at import; override after all imports.
    torch.autograd.set_detect_anomaly(bool(opt.debug or getattr(opt.training, 'detect_anomaly', False)))
    if opt.dist.ddp:
        opt.dist.global_gpu_id = gpu_id
        opt.dist.world_size = opt.dist.ngpus_per_node
        dist.init_process_group(backend='nccl', init_method=opt.dist.dist_url,
                                world_size=opt.dist.world_size, rank=gpu_id,
                                timeout=timedelta(seconds=getattr(opt.training, 'process_group_timeout_seconds', 600)))
    try:
        trainer = SPVNASExperiment(opt)
        if opt.exp_type == 'train':
            trainer.train_epoch()
        else:
            splits = (
                (trainer.syn_known_test_data_loader, 'synthetic-val'),
                (trainer.syn_novel_test_data_loader, 'synthetic-test'),
                (trainer.real_known_test_data_loader, 'real-val'),
                (trainer.real_novel_test_data_loader, 'real-test'),
            )
            for epoch, (loader, name) in enumerate(splits):
                trainer.test_vis_dir = str(Path(trainer.result_dir, '{}_vis'.format(name)))
                Path(trainer.test_vis_dir).mkdir(exist_ok=True)
                trainer.test(loader, 'test', name, epoch=epoch)
    finally:
        if opt.dist.ddp:
            dist.destroy_process_group()
