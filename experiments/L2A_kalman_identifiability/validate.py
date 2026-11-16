import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from kalman import estimate,covariance_joint,initial_covariance,A,Q
R=Path(__file__).parent;O=R/'results';checks={}
reg=json.loads((R/'REGISTERED.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(R/f)==h for f,h in reg['sha256'].items());source=json.loads((R/'SOURCE_MANIFEST.json').read_text());assert all(sha(R/f)==h for f,h in source['files'].items());checks['pretest_and_original_input_hashes']='PASS'
x=np.load(R/'inputs/seed101/X_only.npz')['X'][800:803,:32];mf,ms,pf,ps=estimate(x);c=covariance_joint(32).reshape(96,96);oi=np.arange(32)*3+2;maxf=maxs=maxpf=maxps=0.
for t in range(32):
 rows=np.arange(t*3,t*3+3);ix=oi[:t+1];Cxx=c[np.ix_(ix,ix)];Cyx=c[np.ix_(rows,ix)];coef=np.linalg.solve(Cxx,Cyx.T).T;mm=x[:,:t+1]@coef.T;P=c[np.ix_(rows,rows)]-coef@Cyx.T;maxf=max(maxf,float(np.max(abs(mm-mf[:,t]))));maxpf=max(maxpf,float(np.max(abs(P-pf[t]))));coef=np.linalg.solve(c[np.ix_(oi,oi)],c[np.ix_(rows,oi)].T).T;mm=x@coef.T;P=c[np.ix_(rows,rows)]-coef@c[np.ix_(oi,rows)];maxs=max(maxs,float(np.max(abs(mm-ms[:,t]))));maxps=max(maxps,float(np.max(abs(P-ps[t]))))
assert max(maxf,maxpf,maxs,maxps)<1e-10;checks['independent_block_Gaussian_max_errors']=dict(filter_mean=maxf,smooth_mean=maxs,filter_cov=maxpf,smooth_cov=maxps)
cut=16;fc,_,_,_=estimate(x[:,:cut]);assert np.max(abs(fc-mf[:,:cut]))==0;checks['future_truncation_filter_max_error']=0.
assert np.max(abs(mf[:,:,2]-x))<1e-12 and np.max(abs(ms[:,:,2]-x))<1e-12;assert np.linalg.eigvalsh(pf).min()>-1e-12 and np.linalg.eigvalsh(ps).min()>-1e-12;assert np.linalg.eigvalsh(pf-ps).min()>-1e-12;checks['exact_X_and_covariance_PSD']='PASS'
_,_,pf,ps=estimate(np.zeros((1,500)));floor=np.array([[.41,.1],[.1,.3125]]);assert np.linalg.eigvalsh(pf[19:,:2,:2]-floor).min()>-1e-12;checks['fresh_unseen_innovation_covariance_floor']=floor.tolist();checks['exact_burn_prior_vs_stationary_relative_error']=None
from scipy.linalg import solve_discrete_lyapunov
prior=solve_discrete_lyapunov(A,Q);checks['exact_burn_prior_vs_stationary_relative_error']=float(np.max(abs(initial_covariance()-prior))/np.max(abs(prior)))
d=pd.read_csv(O/'trajectory_metrics.csv.gz');assert len(d)==7200;assert not d.duplicated(['seed','trajectory','task','method']).any();parity=json.loads((O/'L1_parity.json').read_text());checks['L1_replayed_MSE_max_difference']=max(v['max_L1_metric_difference'] for v in parity);checks['records']=len(d);checks['status']='PASS';(O/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
