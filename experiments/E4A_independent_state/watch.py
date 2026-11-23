"""Non-blocking progress records; writes status only, never reads predictions."""
import json,time
from settings import ROOT,OUT
while not (ROOT/'PIPELINE_COMPLETE.json').exists():
 n=len(list(OUT.glob('*/model_*.json')))
 markers=[p.name for p in ROOT.glob('*FROZEN.json')]
 (ROOT/'STATUS.json').write_text(json.dumps(dict(time=time.time(),model_bundles_complete=n,target=800,completed_freezes=markers),indent=2))
 time.sleep(30)
