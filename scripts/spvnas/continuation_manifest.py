"""Content hashes for isolated continuation code, dependencies and official weights."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def files():
    paths=[]
    for directory in ('src','scripts','third_party/spvnas','runs/spvnas_fusion/assets'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                paths.append(path)
    paths.append(ROOT/'runs/spvnas_fusion/weights.json')
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def manifest():
    import json
    mapping=files()
    return dict(signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),files=mapping)


def verify(expected):
    current=manifest()
    if current['signature']!=expected['signature']:
        changed=sorted(k for k in set(current['files'])|set(expected['files'])
                       if current['files'].get(k)!=expected['files'].get(k))
        raise RuntimeError('Continuation source changed: '+str(changed))
    return current
