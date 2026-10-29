"""Read-only environment/data inventory. Writes evidence only inside runs/."""
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import argparse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/spvnas_fusion'

def command(args):
    r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(command=args,returncode=r.returncode,output=r.stdout)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args=parser.parse_args()
    import torch
    evidence=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),base_commit=command(['git','rev-parse','HEAD']),
        spvnas_commit=command(['git','-C',str(ROOT/'third_party/spvnas'),'rev-parse','HEAD']),
        nvcc_system=command(['/usr/bin/nvcc','--version']),
        nvcc_lidf=command(['/home/quchengzou/miniconda3/envs/lidf/bin/nvcc','--version']),
        gpu=command(['nvidia-smi']),dependencies={},datasets={})
    for name in ['torchpack','torchsparse','numba','numpy','opencv-python','open3d','torch-scatter','torchvision','rootpath']:
        try: evidence['dependencies'][name]=metadata.version(name)
        except metadata.PackageNotFoundError: evidence['dependencies'][name]='not installed'
    for name,path in [('train','/ssd/polyu/cleargrasp-dataset-train'),('synthetic-val','/ssd/polyu/cleargrasp-dataset-test-val/synthetic-val')]:
        root=Path(path); images=sorted(root.glob('*/rgb-imgs/*-rgb.jpg')); missing=[]; camera_errors=[]
        for image in images:
            for folder,suffix in [('segmentation-masks','-segmentation-mask.png'),('depth-imgs-rectified','-depth-rectified.exr'),('json-files','-masks.json')]:
                paired=image.parent.parent/folder/image.name.replace('-rgb.jpg',suffix)
                if not paired.is_file(): missing.append(str(paired))
            # Camera schema sampled per object; full pairing checked for all images.
        for obj in sorted(root.iterdir()):
            files=sorted(obj.glob('json-files/*-masks.json'))
            if files:
                meta=json.loads(files[0].read_text())
                try:
                    assert len(meta['camera']['world_pose']['rotation']['quaternion'])==4
                    assert len(meta['camera']['world_pose']['matrix_4x4'])==4
                except (KeyError,AssertionError): camera_errors.append(str(files[0]))
        evidence['datasets'][name]=dict(path=path,images=len(images),missing_pairs=missing,camera_schema_errors=camera_errors,
            per_object={p.name:len(list(p.glob('rgb-imgs/*-rgb.jpg'))) for p in sorted(root.iterdir()) if p.is_dir()})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'environment.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))
    if any(not d['images'] or d['missing_pairs'] or d['camera_schema_errors'] for d in evidence['datasets'].values()):
        raise SystemExit('Dataset mismatch: stop before training')

if __name__=='__main__': main()
