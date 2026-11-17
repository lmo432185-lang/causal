"""Final reproducibility/leakage/timing audit. Run only after final evaluation."""
import json,hashlib,ast
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).parent;OUT=ROOT/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (ROOT/'FINAL_EVALUATION_DONE.json').exists();reg=json.loads((ROOT/'REGISTERED.json').read_text());assert all(sha(ROOT/f)==h for f,h in reg['files'].items());ev=json.loads((ROOT/'EVALUATION_REGISTERED.json').read_text());assert all(sha(ROOT/f)==h for f,h in ev['files'].items());ext=json.loads((ROOT/'EXTENDED_AUDIT_REGISTERED.json').read_text());assert all(sha(ROOT/f)==h for f,h in ext['files'].items());freeze=json.loads((ROOT/'ALL_X_ONLY_FROZEN.json').read_text());supp=json.loads((ROOT/'ALL_SUPPLEMENTARY_AUDIT_FROZEN.json').read_text());assert all(sha(ROOT/f)==h for f,h in freeze['files'].items());assert all(sha(ROOT/f)==h for f,h in supp['files'].items());out={'all_frozen_model_and_decision_hashes':'PASS'}
 backend=json.loads((ROOT/'NATIVE_BACKEND_RECORD.json').read_text());assert all(v for v in backend['same_function_sources'].values());assert all(sha(OUT/case/f)==h for case,files in backend['preexisting_complete_cases'].items() for f,h in files.items());out['preserved_pre_native_cases']=len(backend['preexisting_complete_cases'])
 # Static worker imports and file access: no generator, private configuration, truth files.
 tree=ast.parse((ROOT/'fit_jobs.py').read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)];assert 'config' not in imports and 'generate' not in imports;source=(ROOT/'fit_jobs.py').read_text();assert 'sealed_states' not in source and 'sealed_manifest' not in source;out['X_only_worker_truth_access']='no private module imports or truth-file reads'
 jobs=json.loads((ROOT/'public_jobs.json').read_text());assert len(jobs)==99
 if (ROOT/'positive_public_jobs.json').exists():
  preg=json.loads((ROOT/'POSITIVE_REGISTERED.json').read_text());assert all(sha(ROOT/f)==h for f,h in preg['files'].items());positive=json.loads((ROOT/'POSITIVE_ALL_X_FROZEN.json').read_text());assert all(sha(ROOT/f)==h for f,h in positive['files'].items());assert (ROOT/'POSITIVE_EVALUATION_DONE.json').exists();jobs+=json.loads((ROOT/'positive_public_jobs.json').read_text())
 ncase=len(jobs);opt=[];mirror=[]
 for j in jobs:
  d=OUT/j['case'];m=json.loads((d/'selected_parameters.json').read_text())['models'];assert set(m)=={'G1','G2','G3','G4','single'};dec=json.loads((d/'X_only_decisions.json').read_text());assert len(dec['prediction'])==200 and len(dec['posterior'])==200;assert np.max(abs(np.array(dec['posterior'])[:,1]-np.array(dec['posterior'])[:,3]))<1e-12;mirror.append(dec['mirror_loglik_error']);opt+=json.loads((d/'fit_summary.json').read_text());assert (d/'oracle_results.json').stat().st_mtime>=supp['time'];assert (d/'evaluation_alignment_maps.json').stat().st_mtime>=supp['time']
  for k,p in m.items():
   A=np.array(p['A']);assert np.max(abs(np.linalg.eigvals(A)))<1.;assert p['r']>0
 assert len(opt)==ncase*16;out['recorded_completed_starts']=len(opt);out['mirrored_G4_starts']=ncase*4;out['max_mirrored_loglik_difference']=max(mirror);assert max(mirror)<1e-5
 d=pd.read_csv(OUT/'all_trajectory_results.csv.gz');assert len(d)==ncase*200*5;assert not d.duplicated(['case','trajectory','method']).any();assert (d.correct_declared+d.wrong_declared+d.refused).eq(1).all();sm=pd.read_csv(OUT/'all_hidden_state_metrics.csv.gz');assert len(sm)==ncase*200*3*2*2;assert not sm.duplicated(['case','family','mode','trajectory','state']).any();out['classification_records']=len(d);out['hidden_state_records']=len(sm)
 # Independent theoretical proof of the private E pairs; only now may use true parameters.
 from generate import system
 private=json.loads((ROOT/'data/sealed_manifest.json').read_text());pairs={}
 for meta in private:
  if meta['section']=='E' and meta['seed']==101:pairs.setdefault(meta['tag'],[]).append(meta)
 errors={}
 for tag,metas in pairs.items():
  cov=[]
  for meta in metas:
   A,Q,P=system(meta);v=np.array([(np.linalg.matrix_power(A,k)@P)[2,2] for k in range(101)]);v[0]+=meta['params']['obs']**2;cov.append(v)
  err=float(np.max(abs(cov[0]-cov[1]))/np.max(abs(cov[0])));assert err<1e-10;errors[tag]=err
 out['true_E_covariance_relative_errors']=errors;out['global_X_freeze_time']=freeze['time'];out['supplementary_freeze_time']=supp['time'];out['status']='PASS';(OUT/'audit.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
