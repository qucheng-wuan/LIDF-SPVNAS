"""Tie successful validation to exact integration code/config/artifact hashes."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def signature():
    digest=hashlib.sha256()
    paths=[]
    for pattern in ['src/models/*.py','src/utils/runtime_settings.py','src/trainers/train_spvnas.py','src/experiments/spvnas_fusion/*.yaml','scripts/spvnas/*.py','runs/spvnas_fusion/*60ep*/configs/*.yaml']:
        paths.extend(ROOT.glob(pattern))
    paths.append(ROOT/'runs/spvnas_fusion/weights.json')
    paths.extend((ROOT/'runs/spvnas_fusion/assets/SemanticKITTI_val_SPVNAS@20GMACs').glob('*'))
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
