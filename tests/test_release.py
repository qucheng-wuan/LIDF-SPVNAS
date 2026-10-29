"""CPU checks for distribution paths, validation binding and resume state."""
import copy
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts/spvnas'), str(ROOT / 'src')]
from entry import options
from evidence import configuration_signature


class ReleaseTests(unittest.TestCase):
    def config(self):
        return options([ROOT / 'src/experiments/spvnas_fusion/spvnas_fusion_20g.yaml',
                        ROOT / 'configs/train_60ep.yaml'])

    def test_paths_work_outside_repository_and_data_override(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'LIDF_DATA_ROOT': directory}):
                os.chdir(directory)
                opt = self.config()
                self.assertEqual(opt.dataset.cleargrasp_root_dir, directory)
                self.assertEqual(opt.spvnas.repo_dir, str(ROOT / 'third_party/spvnas'))
                self.assertTrue(Path(opt.base_log_dir).is_absolute())
                self.assertEqual(opt.training.budget_unit, 'epochs')
                self.assertEqual(opt.training.nepochs, 60)
                self.assertIsNone(opt.training.max_steps)
                self.assertEqual(opt.training.micro_batch * 4 * opt.training.accumulation_steps, 32)
        finally:
            os.chdir(previous)

    def test_validation_binding_distinguishes_model_and_output_changes(self):
        opt = self.config()
        before = configuration_signature(opt)
        opt.base_log_dir = '/tmp/new-output'
        opt.resume = '/tmp/epoch.pth'
        opt.training.resume_best_path = '/tmp/best.pth'
        self.assertEqual(before, configuration_signature(opt))
        opt.spvnas.voxel_size = 0.01
        self.assertNotEqual(before, configuration_signature(opt))

    def test_resume_restores_adam_scheduler_and_all_cpu_random_streams(self):
        import numpy as np
        import torch
        from utils.training_resume import capture_runtime, restore_runtime
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        generator = torch.Generator().manual_seed(17)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

        def step():
            value = torch.randn(4, 3) + random.random() + np.random.rand()
            target = torch.randn(4, 2, generator=generator)
            optimizer.zero_grad()
            loss = (model(value) - target).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.detach().clone()

        for _ in range(3):
            step()
        saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
        runtime = capture_runtime(generator)
        expected_losses = [step() for _ in range(3)]
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        scheduler.load_state_dict(saved[2])
        restore_runtime(runtime, generator)
        for loss in expected_losses:
            torch.testing.assert_close(step(), loss, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
