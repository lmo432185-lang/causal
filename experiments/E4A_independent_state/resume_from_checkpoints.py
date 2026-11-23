from pathlib import Path
import shutil,json,hashlib
from settings import ROOT,OUT
OUT.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'fit_checkpoints').glob('*/*.json'):
 q=OUT/p.parent.name/p.name
 if not q.exists():q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
if (ROOT/'MODELS_FROZEN.json').exists():
 for f,h in json.loads((ROOT/'MODELS_FROZEN.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 print('Frozen models restored; continue next stage without refitting')
else:print('Completed models restored; rerun fit.py to finish remaining cases')
