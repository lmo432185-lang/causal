"""Deterministically replay archived generator seeds; does not fit or change frozen outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from generate import generate_one
R=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--out',default='/tmp/L4_replay');a=p.parse_args()
items=json.loads((R/'data/sealed_manifest.json').read_text())
f=R/'data/positive_sealed_manifest.json'
if f.exists():
 v=json.loads(f.read_text());items+=v if isinstance(v,list) else v.get('cases',[])
m=next((x for x in items if x['case']==a.case),None)
if m is None:raise SystemExit('Unknown case; see data/*manifest.json')
d=Path(a.out)/a.case
if d.exists():raise SystemExit('Output directory already exists; choose a new --out to preserve results')
d.mkdir(parents=True);x,s,meta=generate_one(m,m['seed']);np.save(d/'X.npy',x);np.savez_compressed(d/'sealed_states.npz',physical=s);(d/'generation.json').write_text(json.dumps(meta,indent=2));print(d)
