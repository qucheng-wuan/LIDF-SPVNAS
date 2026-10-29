"""Stable tie handling for the original ray-wise maximum selection."""
import torch
from torch_scatter import scatter_max


def stable_ray_argmax(scores, ray_ids, ray_count):
    """Same max score; equal maxima choose first voxel in canonical grid order.

    torch_scatter's CUDA argmax may select different ties depending on thread
    scheduling. The return sentinel len(scores) preserves LIDF's dummy zero point
    for rays without an intersection. No logits/probabilities are perturbed. For inference callers pass logits
    directly, using softmax monotonicity without its float32 rounding ties.
    """
    maxima, _ = scatter_max(scores, ray_ids, dim_size=ray_count)
    indices = torch.arange(len(scores), device=scores.device)
    candidates = torch.where(scores == maxima[ray_ids], indices, len(scores))
    result = torch.full((ray_count,), len(scores), dtype=torch.long, device=scores.device)
    result.scatter_reduce_(0, ray_ids, candidates, reduce='amin', include_self=True)
    return result
