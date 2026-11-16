import json,hashlib
import numpy as np,pandas as pd
from pathlib import Path
from core import plant,kalman,covariance_schedule
from config import CELLS,SEEDS
R=Path(__file__).parent;O=R/'results';reg=json.loads((R/'REGISTERED.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(R/f)==h for f,h in {**reg['files'],**reg['inputs']}.items());checks={'all_registered_code_and_inputs':'PASS'}
# Independent joint-Gaussian conditioning with nonzero observation noise.
A,Q,obsR=plant(**CELLS[5]['params']);T=25;p=np.zeros_like(A)
for _ in range(201):p=A@p@A.T+Q
C=np.empty((T,3,T,3));Pt=[]
for t in range(T):
 if t:p=A@p@A.T+Q
 Pt.append(p.copy())
for t in range(T):
 for k in range(t+1):C[t,:,k,:]=np.linalg.matrix_power(A,t-k)@Pt[k];C[k,:,t,:]=C[t,:,k,:].T
C=C.reshape(T*3,T*3);obs=np.arange(T)*3+2;Cxx=C[np.ix_(obs,obs)]+obsR*np.eye(T);x=np.random.default_rng(17).normal(size=(3,T));mf,ms,pf,ps=kalman(x,A,Q,obsR);err=0.
for t in range(T):
 rows=np.arange(t*3,t*3+3);ix=obs[:t+1];coef=np.linalg.solve(Cxx[:t+1,:t+1],C[np.ix_(rows,ix)].T).T;err=max(err,float(np.max(abs(x[:,:t+1]@coef.T-mf[:,t]))));coef=np.linalg.solve(Cxx,C[np.ix_(rows,obs)].T).T;err=max(err,float(np.max(abs(x@coef.T-ms[:,t]))))
assert err<1e-10;checks['independent_Gaussian_filter_smoother_error']=err;cut=10;f,_,_,_=kalman(x[:,:cut],A,Q,obsR);assert np.max(abs(f-mf[:,:cut]))==0;checks['future_truncation_error']=0.
d=pd.read_csv(O/'scan_metrics.csv.gz');assert len(d)==66000;assert not d.duplicated(['cell','seed','trajectory','state','method']).any();assert d.groupby(['cell','seed','state','method']).size().eq(200).all();checks['scan_metrics']=len(d)
for cell in CELLS:
 for seed in SEEDS:
  dr=O/'scan'/cell['name']/str(seed);fr=json.loads((dr/'FREEZE.json').read_text());assert sha(dr/'matched.pt')==fr['checkpoint_sha256']
st=pd.read_csv(O/'structure_state_metrics.csv.gz');assert len(st)==3000;assert not st.duplicated(['group','seed','trajectory','state']).any();checks['structure_metrics']=len(st);models=json.loads((O/'structure_models.json').read_text());assert all(m['causal_refused'] for m in models);checks['generic_models']=len(models)
p=json.loads((O/'equivalence_proofs.json').read_text());assert len(p)==3 and all(v['covariance_max_relative_error']<1e-10 and v['spectrum_max_relative_error']<1e-10 for v in p);e=pd.read_csv(O/'equivalence_metrics.csv.gz');assert len(e)==3600 and e.refused.all();checks['equivalence_metrics']=len(e);checks['max_exact_covariance_error']=max(v['covariance_max_relative_error'] for v in p);checks['status']='PASS';(O/'audit.json').write_text(json.dumps(checks,indent=2));print(checks)
jreg=json.loads((R/'JOINT_DIAGNOSTIC_REGISTERED.json').read_text());assert sha(R/'diagnostic_joint.py')==jreg['sha256'];joint=pd.read_csv(O/'posthoc_joint_metrics.csv.gz');assert len(joint)==9000 and not joint.duplicated(['group','seed','trajectory','state','method']).any();checks['separately_registered_posthoc_joint_metrics']=len(joint);(O/'audit.json').write_text(json.dumps(checks,indent=2))
