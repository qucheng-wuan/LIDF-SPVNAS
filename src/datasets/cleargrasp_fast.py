"""Equivalent deterministic preprocessing, used only by the continuation trainer."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.cleargrasp_synthetic_dataset import (
    ClearGrasp_Syn_Object_Dataset, connected_components)
from utils import data_augmentation, seg_utils


@lru_cache(maxsize=2)
def pixel_indices(height, width):
    indices = seg_utils.build_matrix_of_indices(height, width)
    indices.setflags(write=False)
    return indices


def compute_xyz_cached(depth, camera):
    indices = pixel_indices(camera['yres'], camera['xres'])
    # Preserve original dtype and operation order; cache only the pixel grid.
    x = (indices[..., 1] - camera['cx']) * depth / camera['fx']
    y = (indices[..., 0] - camera['cy']) * depth / camera['fy']
    return np.stack([x, y, depth], axis=-1)


class FastClearGraspSynthetic(ClearGrasp_Syn_Object_Dataset):
    def process_depth(self, depth_filename, camera_params, corrupt_mask_float):
        if self.exp_type == 'train' and self.params['depth_aug']:
            return super().process_depth(depth_filename, camera_params, corrupt_mask_float)
        depth = data_augmentation.exr_loader(depth_filename, 1)
        xyz = compute_xyz_cached(depth, camera_params)
        size = (self.params['img_width'], self.params['img_height'])
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        xyz = cv2.resize(xyz, size, interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        xyz = torch.from_numpy(xyz).permute(2, 0, 1).float()
        return depth, xyz, depth * (1 - corrupt_mask_float), xyz * (1 - corrupt_mask_float)

    def process_label(self, mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        labels, _ = connected_components(mask == 255)
        # scipy labels components 1..N. Original unique+enumerate is identity
        # if background 0 exists; all-foreground maps its only label 1 to 0.
        labels -= labels.min()
        return cv2.resize(labels, (self.params['img_width'], self.params['img_height']),
                          interpolation=cv2.INTER_NEAREST)


def get_dataset(dataset_dir, params, exp_type, obj_type='known'):
    if exp_type == 'train':
        subdir = Path(dataset_dir) / 'cleargrasp-dataset-train'
    elif exp_type in ('valid', 'test'):
        split = 'synthetic-test' if obj_type == 'novel' else 'synthetic-val'
        subdir = Path(dataset_dir) / 'cleargrasp-dataset-test-val' / split
    else:
        raise ValueError(exp_type)
    params = params.copy()
    if exp_type != 'train':
        params['use_data_augmentation'] = False
    return FastClearGraspSynthetic([str(subdir)], exp_type, params)
