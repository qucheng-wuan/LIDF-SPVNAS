"""Download only official specialized artifacts; never substitute a model."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
# Official legacy host (SPVNAS issue: change hanlab to hanlab18).
BASE = 'https://hanlab18.mit.edu/files/SPVNAS/spvnas_specialized/'

def main():
    records = []
    for size in (20, 65):
        name = f'SemanticKITTI_val_SPVNAS@{size}GMACs'
        for filename in ('net.config', 'init'):
            path = ROOT / 'runs/spvnas_fusion/assets' / name / filename
            url = BASE + name + '/' + filename
            record = dict(url=url, path=str(path))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with urlopen(url, timeout=45) as response:
                        data = response.read()
                    if filename == 'net.config':
                        json.loads(data)
                    temporary = path.with_suffix('.partial')
                    temporary.write_bytes(data)
                    temporary.replace(path)
                record.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception as exc:
                record['error'] = repr(exc)
            print(json.dumps(record), flush=True)
            records.append(record)
    (ROOT / 'runs/spvnas_fusion/weights.json').write_text(json.dumps(records, indent=2))
    if any('error' in r for r in records):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
