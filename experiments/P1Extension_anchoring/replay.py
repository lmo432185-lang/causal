"""Regenerate a frozen test probability file from saved parameters without fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from settings import *
from model import infer
from private_generator import paired

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--out',required=True);args=p.parse_args();target=Path(args.out)
 if target.exists():raise SystemExit('Choose a new output directory; existing outputs are preserved')
 meta=next(m for m in json.loads((DATA/'sealed_manifest.json').read_text()) if m['case']==args.case);pair,seed=paired(meta,meta['seed']);s,x,I=pair[meta['policy']];target.mkdir(parents=True);out={}
 for m in METHODS:
  r=json.loads((OUT/args.case/f'{m}_models.json').read_text());a=r['selected'];kind=r.get('effective_model','E0');theta=np.array(a['theta']);d=a['delay'];v=infer(x[-TEST:],theta,d,kind,I[-TEST:] if kind=='E3' else None)
  for k1,k2 in [('mean','mean'),('cov','cov'),('forecast','forecast'),('forecast_var','forecast_var'),('test_LL','trajectory_ll')]:out[m+'_'+k1]=v[k2]
  if m in ['E0','E1','E2'] or (m=='E3' and meta['policy']=='passive'):
   for dd in [2,3]:
    v=infer(x[-TEST:],theta,dd,kind);out[m+f'_d{dd}_mean']=v['mean'];out[m+f'_d{dd}_cov']=v['cov']
 np.savez_compressed(target/'permitted_predictions.npz',**out);wanted=json.loads((OUT/args.case/'LOCAL_X_FREEZE.json').read_text())['files']['permitted_predictions.npz'];actual=hashlib.sha256((target/'permitted_predictions.npz').read_bytes()).hexdigest();result=dict(case=args.case,full_probability_output_bitwise_matches=actual==wanted,actual_sha256=actual,expected_sha256=wanted,refitted=False);(target/'REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2));assert actual==wanted,result;print('Frozen replay PASS',args.case)
if __name__=='__main__':main()
