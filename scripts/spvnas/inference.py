"""GT-free Stage-1 inference using the unchanged LIDF geometry and decoder."""
import torch

@torch.no_grad()
def predict(model, batch):
    # prepare_data historically requires 'xyz'; its flattened copy is used only
    # by compute_gt/compute_loss, which this inference path never calls.
    inputs={k:v for k,v in batch.items() if k not in ('xyz','depth')}
    inputs['xyz']=inputs['xyz_corrupt']
    values=model.prepare_data(inputs,'test',None)
    model.get_valid_points(values)
    if not model.get_occ_vox_bound(values):
        raise RuntimeError('No occupied voxels')
    model.get_miss_ray(values,'test')
    if not values['total_miss_sample_num'] or not model.compute_ray_aabb(values):
        raise RuntimeError('No ray intersections')
    model.get_embedding(values)
    model.get_pred(values,'test',0)
    return values['pred_pos']
