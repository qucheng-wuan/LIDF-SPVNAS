"""Short real-trainer diagnostic, optionally preloading data or tracing rank zero.

Run with torchrun and a debug overlay restricting smoke_train_images/nepochs.
Outputs are separate from formal training. --source-tree supports before/after
source snapshots while compiling identical CUDA extensions from the real root.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-tree', type=Path, default=ROOT)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_tree.resolve() / 'src'))
    from opt import Params
    opt = Params(str(ROOT / 'src/experiments/implicit_depth/default_config.yaml'))
    for path in args.configs:
        opt.update(path)
    if not opt.debug or not 0 < opt.training.smoke_train_images <= 1024 or opt.training.nepochs != 1:
        raise ValueError('Benchmark requires a bounded debug subset and exactly one epoch')
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert opt.training.micro_batch * world * opt.training.accumulation_steps == opt.training.batch_size
    opt.dist.ddp = True
    opt.dist.ngpus_per_node = world
    opt.dist.dist_url = 'env://'
    os.chdir(ROOT / 'src')
    import torch
    from trainers.train_spvnas import SPVNASExperiment, createAndRunTrainer
    if args.profile and rank == 0:
        from models.pipeline import LIDF
        from models.spvnas_fusion import SPVNASFusion
        def instrument(cls, name):
            original = getattr(cls, name)
            def wrapped(*a, **kw):
                with torch.profiler.record_function(f'{cls.__name__}.{name}'):
                    return original(*a, **kw)
            setattr(cls, name, wrapped)
        for name in ('prepare_data', 'get_valid_points', 'get_occ_vox_bound', 'get_miss_ray',
                     'compute_ray_aabb', 'compute_gt', 'get_embedding', 'get_pred', 'compute_loss'):
            instrument(LIDF, name)
        instrument(SPVNASFusion, 'forward')
    original_setup_data = SPVNASExperiment.setup_data
    original_train = SPVNASExperiment.train_epoch

    def setup_data(self):
        original_setup_data(self)
        if args.preload:
            self.train_data_loader = list(self.train_data_loader)

    def train_epoch(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            initial_step = self.optimizer.step
            captured = False
            def audit_step(*a, **kw):
                nonlocal captured
                if not captured:
                    torch.save({name: p.grad.detach().cpu() for name, p in self.network.named_parameters()
                                if p.grad is not None}, Path(opt.base_log_dir, 'first_update_gradients.pth'))
                    captured = True
                return initial_step(*a, **kw)
            self.optimizer.step = audit_step
        if args.profile and rank == 0:
            def trace_ready(prof):
                out = Path(opt.base_log_dir)
                (out / 'profile_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))
                (out / 'profile_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
                (out / 'profile_stages.json').write_text(json.dumps([
                    {'name': event.key, 'count': event.count, 'cpu_total_us': event.cpu_time_total,
                     'cuda_total_us': event.cuda_time_total}
                    for event in prof.key_averages() if event.key.startswith(('LIDF.', 'SPVNASFusion.'))], indent=2))
                prof.export_chrome_trace(str(out / 'profile.json'))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
                    on_trace_ready=trace_ready) as prof:
                original_step = self.optimizer.step
                def step(*a, **kw):
                    result = original_step(*a, **kw)
                    prof.step()
                    return result
                self.optimizer.step = step
                original_train(self)
        else:
            original_train(self)
        Path(opt.base_log_dir, f'memory_rank{rank}.json').write_text(json.dumps({
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'preloaded': args.preload}))

    SPVNASExperiment.setup_data = setup_data
    SPVNASExperiment.train_epoch = train_epoch
    createAndRunTrainer(rank, opt)


if __name__ == '__main__':
    main()
