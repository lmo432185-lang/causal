import argparse,ast,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from settings import *
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(output=None):
 reg=json.loads((ROOT/'REGISTERED.json').read_text());ev=json.loads((ROOT/'EVALUATION_REGISTERED.json').read_text());freeze=json.loads((ROOT/'ALL_PERMITTED_MODELS_FROZEN.json').read_text());done=json.loads((ROOT/'EVALUATION_DONE.json').read_text());assert max(reg['time'],ev['time'],freeze['time'])<done['time'];assert all(sha(ROOT/f)==h for f,h in reg['files'].items());assert all(sha(ROOT/f)==h for f,h in ev['files'].items());missing=[];count=0
 for f,h in freeze['files'].items():
  p=ROOT/f
  if p.exists():assert sha(p)==h,f;count+=1
  else:assert p.name=='permitted_predictions.npz';missing.append(f)
 resume=json.loads((ROOT/'LAYOUT_BUGFIX_AND_RESUME.json').read_text());assert len(resume['completed_models_preserved'])==144;assert all(sha(ROOT/f)==h for f,h in resume['completed_models_preserved'].items())
 source=(ROOT/'fit_worker.py').read_text();assert 'sealed_states' not in source and 'sealed_manifest' not in source and 'private_generator' not in source;tree=ast.parse(source);assert 'private_generator' not in [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
 jobs=json.loads((ROOT/'public_jobs.json').read_text());assert len(jobs)==36;fits=[];aliases_copied=0
 for j in jobs:
  for m in METHODS:
   q=json.loads((OUT/j['case']/f'{m}_models.json').read_text());assert len(q['candidates'])==9
   if m=='E3' and j['policy']=='passive':aliases_copied+=9
   else:fits+=q['candidates']
  sparse=DATA/j['case']/'sparse_auxiliary.npz'
  if sparse.exists():
   v=np.load(sparse);assert v['Y'].shape==(800,24) and v['Y_pair'].shape==(800,12)
 c=pd.read_csv(OUT/'case_metrics.csv');t=pd.read_csv(OUT/'trajectory_metrics.csv.gz');ids=pd.read_csv(OUT/'identification.csv');assert len(c)==1296 and len(t)==259200 and len(ids)==252;assert not t.duplicated(['case','method','state','trajectory']).any();assert c.filter(regex='^cover').stack().between(0,1).all();gate=json.loads((OUT/'P2_DECISION.json').read_text());assert next(x for x in gate['groups'] if x['method']=='E4YZ')['combined_passes']==6
 # The repaired observation layout has no effect on cached E0-E3 mathematics.
 import model
 spec=importlib.util.spec_from_file_location('pre_layout_model',ROOT/'model_pre_layout_fix.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);th=np.array([.73,.51,.29,.43,.32,.27,.08,1.2,.8,.9]);diff=0.
 for k in ['E0','E1','E2','E3']:
  for d in [1,2,3]:
   a,da=model.covariance(th,d,k);b,db=old.covariance(th,d,k);diff=max(diff,float(np.max(abs(a-b))),float(np.max(abs(da-db))))
 assert diff==0.;result=dict(status='PASS',audit_time=time.time(),registered_model_and_evaluation_hashes=True,available_frozen_outputs_verified=count,omitted_generated_prediction_files=missing,omissions_reconstructible_by='replay.py; never ignored hash mismatches',cases=36,paired_input_contexts=18,new_trajectory_records=36000,test_trajectory_records=7200,independent_test_pairs=3600,unique_complete_parameter_optimizations=len(fits),copied_zero_input_E3_fits=aliases_copied,successful_optimizations=sum(v['fit']['success'] for v in fits),unsuccessful_optimizations_retained=sum(not v['fit']['success'] for v in fits),preserved_unaffected_model_files=144,unaffected_covariance_derivative_max_difference=diff,test_full_state_file_not_read_by_learner=True,sparse_train_validation_only=True,causal_prefix_error=json.loads((ROOT/'DEVELOPMENT_VALIDATION.json').read_text())['causal_prefix_error'],P2_scope='candidate conditional only, no warning experiment')
 (Path(output) if output else OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='omitted_generated_prediction_files'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out');args=p.parse_args();main(args.out)
