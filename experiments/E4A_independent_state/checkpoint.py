"""Byte-preserving backup of completed numerical models; no fitting or data access."""
import json,time,hashlib
from settings import ROOT,OUT
backup=ROOT/'fit_checkpoints';backup.mkdir(exist_ok=True)
while True:
 for p in OUT.glob('*/*.json'):
  if not (p.name.startswith('model_') or p.name=='INPUT_HASHES.json'):continue
  q=backup/p.parent.name/p.name
  if q.exists():continue
  try:
   data=p.read_bytes();json.loads(data);q.parent.mkdir(exist_ok=True);q.write_bytes(data)
  except (ValueError,OSError):continue
 if (ROOT/'MODELS_FROZEN.json').exists():
  m=json.loads((ROOT/'MODELS_FROZEN.json').read_text());ok=all(hashlib.sha256((backup/f.split('/',1)[1]).read_bytes()).hexdigest()==h for f,h in m['files'].items());assert ok;(ROOT/'CHECKPOINT_AUDIT.json').write_text(json.dumps(dict(completed_models=len(m['files']),byte_identical=True),indent=2));break
 time.sleep(20)
