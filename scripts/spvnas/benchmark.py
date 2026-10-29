"""Isolated GT-free inference benchmark: no optimizer/training tensors resident."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from entry import options
from project import setup_environment
from launch import check_idle


def main():
    p=argparse.ArgumentParser()
    p.add_argument('variant',choices=['original','spvnas_fusion_20g','spvnas_fusion_65g'])
    p.add_argument('--gpu',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--overlay',action='append',default=[])
    a=p.parse_args()
    setup_environment()
    hardware=check_idle([a.gpu])
    checkpoint_path=Path(a.checkpoint).resolve()
    output=Path(a.output).resolve()
    if output.exists(): raise FileExistsError(output)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CUDA_HOME=os.environ.get('CUDA_HOME', sys.prefix),
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        TORCH_EXTENSIONS_DIR=os.environ['TORCH_EXTENSIONS_DIR'])
    import torch
    from torch.utils.data import DataLoader
    from utils.ipc_runtime import check_temporary_ipc
    check_temporary_ipc()
    from datasets.cleargrasp_synthetic_dataset import get_dataset
    from inference import predict
    opt=options([ROOT/f'src/experiments/spvnas_fusion/{a.variant}.yaml', *a.overlay])
    opt.gpu_id=0;opt.dist.ddp=False;opt.grid.valid_sample_num=-1
    opt.base_log_dir=str(output.parent)
    os.chdir(ROOT/'src')
    from models.pipeline import LIDF
    torch.cuda.set_device(0)
    from utils.runtime_settings import configure_backends
    runtime_settings = configure_backends(opt)
    model=LIDF(opt,torch.device('cuda:0'))
    checkpoint=torch.load(checkpoint_path,map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model.eval()
    batch=next(iter(DataLoader(get_dataset(opt.dataset.cleargrasp_root_dir,opt.dataset.dict.copy(),'test','known'),batch_size=1)))
    with torch.no_grad():
        for _ in range(10): predict(model,batch)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(50): predict(model,batch)
        torch.cuda.synchronize()
    record=dict(variant=a.variant,checkpoint=a.checkpoint,hardware=hardware,input=batch['item_path'],
        batch_size=1,height=240,width=320,warmup=10,iterations=50,mean_seconds=(time.perf_counter()-start)/50,
        peak_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
        scope='GT-free Stage 1: input transfer, geometry, both branches and prediction head')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
