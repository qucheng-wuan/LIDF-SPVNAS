"""Shared numerical settings for training, validation and inference timing."""
import torch


def configure_backends(opt):
    import cv2
    cv2.setNumThreads(1)
    deterministic = getattr(opt.training, 'deterministic', True)
    allow_tf32 = getattr(opt.training, 'allow_tf32', False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    return dict(deterministic=deterministic, cudnn_benchmark=not deterministic,
                allow_tf32=allow_tf32, sparse_allow_tf32=getattr(opt.spvnas, 'allow_tf32', False))
