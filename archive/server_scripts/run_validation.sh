#!/usr/bin/env bash
set -uo pipefail
cd /home/quchengzou/implicit_depth
py=/home/quchengzou/miniconda3/envs/lidf/bin/python
"$py" scripts/spvnas/validate.py original --gpu 0 > runs/spvnas_fusion/smoke_original.log 2>&1 &
p0=$!
"$py" scripts/spvnas/validate.py spvnas_fusion_20g --gpu 1 > runs/spvnas_fusion/smoke_spvnas_fusion_20g.log 2>&1 &
p20=$!
"$py" scripts/spvnas/validate.py spvnas_fusion_65g --gpu 4 > runs/spvnas_fusion/smoke_spvnas_fusion_65g.log 2>&1 &
p65=$!
failed=0
wait "$p0" || failed=1
wait "$p20" || failed=1
wait "$p65" || failed=1
"$py" - <<'PY'
import json
from pathlib import Path
for v in ['original','spvnas_fusion_20g','spvnas_fusion_65g']:
 d=json.loads(Path(f'runs/spvnas_fusion/validation/{v}.json').read_text())
 print(v,d['status'],d.get('checks'),d.get('error',''),flush=True)
PY
exit "$failed"
