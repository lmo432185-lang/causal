"""Integrity and leakage audit. Missing generated arrays in Git bundle are explicitly reported."""
import json,hashlib,time,ast,argparse
from pathlib import Path
import numpy as np,pandas as pd
from settings import ROOT,OUT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(output=None):
 reg=json.loads((ROOT/'REGISTERED.json').read_text());ev=json.loads((ROOT/'EVALUATION_REGISTERED.json').read_text());freeze=json.loads((ROOT/'ALL_X_ONLY_FROZEN.json').read_text());done=json.loads((ROOT/'EVALUATION_DONE.json').read_text());assert reg['time']<freeze['time']<ev['time']<done['time'];assert all(sha(ROOT/f)==h for f,h in reg['files'].items());assert all(sha(ROOT/f)==h for f,h in ev['files'].items());missing=[];checked=0
 for f,h in freeze['files'].items():
  p=ROOT/f
  if p.exists():assert sha(p)==h,f;checked+=1
  else:
   assert p.name=='X_only_predictions.npz',f;missing.append(f)
 imports=[n.module for n in ast.walk(ast.parse((ROOT/'fit_worker.py').read_text())) if isinstance(n,ast.ImportFrom)];assert 'private_generator' not in imports
 # Explicit I/O barrier: worker/encoder/model contain no read of private answers.
 for f in ['fit_worker.py','encoder.py','model.py']:
  s=(ROOT/f).read_text();assert 'sealed_states' not in s and 'sealed_manifest' not in s and 'private_generator' not in s.replace('does not import private_generator','')
 jobs=json.loads((ROOT/'public_jobs.json').read_text());assert len(jobs)==39;opts=[];boots=[];llerr=[]
 for j in jobs:
  d=OUT/j['case'];p=json.loads((d/'parameters.json').read_text());opts+=p['fits'];boots+=json.loads((d/'bootstrap.json').read_text())['replicates'];L=p['delay_trajectory_LL'];llerr += [float(np.max(abs(np.array(L[str(k)])-np.array(L['1'])))) for k in [2,3]];assert (d/'evaluation_examples.npz').stat().st_mtime>=freeze['time']
 assert max(llerr)<1e-8;assert len(opts)==156 and len(boots)==108
 c=pd.read_csv(OUT/'case_metrics.csv');t=pd.read_csv(OUT/'trajectory_metrics.csv.gz');assert len(c)==1470 and len(t)==294000;assert not t.duplicated(['case','method','state','trajectory']).any();assert len(c[c.state=='X_next'])==156;assert (c.filter(regex='^cover').stack().between(0,1)).all();g=json.loads((OUT/'P2_DECISION.json').read_text());assert g['conditional_passed']==0 and not g['physical_identity_gate'] and not g['delay_gate'];o=pd.read_csv(OUT/'observability.csv');assert (o.observability_rank==o.augmented_dimension).all();assert o.delay_X_covariance_max_difference.max()<1e-9;assert len(list((OUT/'figures').glob('*.png')))==10
 ctr=json.loads((OUT/'identifiability_controls.json').read_text());assert ctr['scale_gauge_covariance_max_difference']<1e-10
 result=dict(status='PASS',scope='present registered files, archived weights and metrics; generated-output omissions disclosed',registered_hashes_verified=True,available_frozen_output_files_verified=checked,omitted_generated_prediction_files=missing,omitted_files_reconstructible_by='replay.py with sealed generation manifest; not ignored hash mismatches',datasets=39,new_trajectories=39000,test_trajectories=7800,main_initializations=len(opts),successful_main_initializations=sum(v['success'] for v in opts),bootstrap_fits=len(boots),successful_bootstrap_fits=sum(v['fit']['success'] for v in boots),probability_record_count=len(t),max_compatible_delay_LL_difference=max(llerr),truth_free_worker_static_checks=True,freeze_before_evaluation=True,evaluation_code_unchanged=True,causal_prefix_test=json.loads((ROOT/'DEVELOPMENT_VALIDATION.json').read_text())['causal_prefix_error'],P2_gate='not passed',audit_time=time.time())
 dest=Path(output) if output else OUT/'audit.json';dest.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='omitted_generated_prediction_files'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out');a=p.parse_args();main(a.out)
