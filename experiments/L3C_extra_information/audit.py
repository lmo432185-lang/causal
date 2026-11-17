import json,hashlib
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal
from config import *
from physics import *
reg=json.loads((R/'REGISTERED.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(R/f)==h for f,h in reg['files'].items());out={'pretest_hashes':'PASS'}
# Independent dense Gaussian likelihood, including nonzero controlled mean and vector observation.
maxerr=0.
for info in ['C1_XZ','C2_Y']:
 p=PAIRS[0]['models'][1][1];sigma=.25;A,Q=matrices(p);P=solve_discrete_lyapunov(A,Q);H=observation(info);L=80;dim=len(H);obs,_=simulate(p,314159,sigma,info,3);cov=np.empty((L,dim,L,dim))
 for t in range(L):
  for k in range(t+1):cov[t,:,k,:]=H@np.linalg.matrix_power(A,t-k)@P@H.T;cov[k,:,t,:]=cov[t,:,k,:].T
 C=cov.reshape(L*dim,L*dim)+np.eye(L*dim)*sigma*sigma;mu=controlled_mean(p,info)[:L].ravel();dense=multivariate_normal.logpdf(obs[:,:L].reshape(3,-1),mean=mu,cov=C);ll=log_likelihood(obs,p,sigma,info,[80,500])[80];maxerr=max(maxerr,float(np.max(abs(dense-ll))));assert np.max(abs(dense-ll))<1e-8
out['dense_Gaussian_vs_Kalman_loglik_max_error']=maxerr
# Prefix score is unchanged by editing all future observations.
p=PRIMARY[1][1];obs,st=simulate(p,3141,.25,'C2_Y',3);orig=log_likelihood(obs,p,.25,'C2_Y',[100])[100];changed=obs.copy();changed[:,100:]+=100.;cut=log_likelihood(changed,p,.25,'C2_Y',[100])[100];assert np.max(abs(orig-cut))==0;out['future_perturbation_prefix_loglik_error']=0.
# Same innovations, same measured X between C0 and joint-sensor condition.
x,s0=simulate(p,123,.25,'C0_X',3);xz,s1=simulate(p,123,.25,'C1_XZ',3);assert np.max(abs(s0-s1))==0 and np.max(abs(x[:,:,0]-xz[:,:,0]))==0;out['paired_X_and_states_equal']='PASS'
proof=json.loads((O/'pretest_law_comparisons.json').read_text());q=pd.DataFrame(proof['rows']);expect={PAIRS[0]['name']:{'C0_X','C2_X_negative','C3_constraints_X'},PAIRS[1]['name']:{'C0_X','C2_Y','C2_X_negative','C3_constraints_X'},PAIRS[2]['name']:{'C0_X','C1_XZ','C2_Y','C2_X_negative','C3_constraints_X'}}
for pair,same in expect.items():assert set(q[(q.pair==pair)&q.observationally_equivalent]['info'])==same
out['predeclared_information_law_patterns']='PASS';d=pd.read_csv(O/'trajectory_metrics.csv.gz');assert len(d)==201600;assert not d.duplicated(['pool','truth_idx','seed','trajectory','noise','info','length']).any();assert (d.correct_declared+d.wrong_declared+d.refused).eq(1).all();assert d.groupby(['pool','truth_idx','seed','noise','info','length']).size().eq(200).all()
for pi,pair in enumerate(PAIRS,1):
 for inf in expect[pair['name']]-{'C3_constraints_X'}:assert d[(d.pool==pi)&(d['info']==inf)].refused.eq(1).all()
invalid=d[(d['info']=='C3_constraints_X')&~d.prior_valid];assert invalid.wrong_declared.eq(1).all();weak=d[d.pool==4];assert weak.refused.eq(1).all();out['rows']=len(d);out['invalid_prior_trials_retained']=len(invalid);out['weak_prior_still_ambiguous']=True;out['status']='PASS';(O/'audit.json').write_text(json.dumps(out,indent=2));print(out)
# Verify the auxiliary physical Z channels preserve every original L3-B X law.
source=json.loads((R/'inputs/SOURCE.json').read_text());oldfile=R/'inputs/L3B_original_equivalence_proofs.json';assert sha(oldfile)==source['sha256'];old=json.loads(oldfile.read_text());parity=0.
for pair,orig in zip(PAIRS,old):
 for pp,oldp in zip([m[1] for m in pair['models']],[orig['H0'],orig['H1']]):
  if oldp['kind']=='direct':AA=np.array([[oldp['a'],0.],[oldp['beta'],.5]]);QQ=np.diag([oldp.get('qy',.25),oldp.get('qx',.25)])
  else:AA=np.array([[oldp['a'],0.,0.],[oldp['alpha'],oldp['c'],0.],[oldp.get('beta1',0.),oldp['beta'],.5]]);QQ=np.diag([oldp.get('qy',.25),oldp.get('qz',.25),oldp.get('qx',.25)])
  PP=solve_discrete_lyapunov(AA,QQ);oldcov=np.array([(np.linalg.matrix_power(AA,k)@PP)[-1,-1] for k in range(101)]);newcov=kernel(pp,'C0_X',0.)[:,0,0];parity=max(parity,float(np.max(abs(oldcov-newcov))))
assert parity<1e-10;out['original_L3B_X_covariance_max_difference']=parity;(O/'audit.json').write_text(json.dumps(out,indent=2))
