from pathlib import Path
import shutil,json,time,hashlib
from config import *
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 src=Path('/workspace/work/L2A/inputs');inp=R/'inputs';inp.mkdir(exist_ok=True);sources={}
 for seed in SEEDS:
  d=inp/str(seed);d.mkdir(exist_ok=True)
  for name in ['X_only.npz','sealed_truth.npz','frozen.pt','probe_maps.json']:
   a=src/f'seed{seed}'/name;b=d/name;shutil.copy2(a,b);assert sha(a)==sha(b);sources[str(b.relative_to(R))]=sha(b)
 (R/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2));files=['main.py','config.py','core.py','encoder.py','scan.py','predictive.py','structure.py','equivalence.py','PROTOCOL.json'];(R/'REGISTERED.json').write_text(json.dumps(dict(time=time.time(),files={f:sha(R/f) for f in files},inputs=sources,source_commit='a033238a17ccab47c417ea2f03e6da4f5bf0f9e6',source='L1 immutable archives via L2A inputs'),indent=2))
 from scan import run as scan
 from structure import run as structure
 from equivalence import run as equivalent
 scan();structure();equivalent();print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
