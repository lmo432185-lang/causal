"""Truth audit, accessible only after both final prediction tests are frozen."""
import json,time,hashlib,numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from model import unpack
from producer import generate,matrices,SYSTEMS
import native

def quality(mu,cov,y):
 mu=mu[:,200:];v=np.diagonal(cov[200:],axis1=1,axis2=2);y=y[:,200:];err=y-mu;dim=y.shape[-1];ans=[]
 for k in range(dim):
  vv=np.maximum(v[:,k],1e-12);e=err[:,:,k];corr=np.corrcoef(mu[:,:,k].ravel(),y[:,:,k].ravel())[0,1];nll=np.mean(.5*(np.log(2*np.pi*vv)+e*e/vv));ans.append(dict(coordinate=k,corr=corr,MSE=np.mean(e*e),NLL=nll,cover90=np.mean(abs(e)<=1.6448536269514722*np.sqrt(vv)),width90=np.mean(2*1.6448536269514722*np.sqrt(vv))))
 inv=np.linalg.pinv(cov[200:]);mah=np.einsum('nti,tij,ntj->nt',err,inv,err);joint=float(np.mean(mah<=chi2.ppf(.9,dim)));pcorr=float(np.mean(cov[200:,0,1]/np.sqrt(v[:,0]*v[:,1]))) if dim==2 else np.nan
 for r in ans:r.update(joint_cover90=joint,posterior_corr=pcorr)
 return ans

def fitted_post(x,m):
 p=native.run(x/m['scale'],m);return p['mean']*m['scale'],p['cov']*m['scale']**2

def work(j):
 assert (ROOT/'TEST_A_FROZEN.json').exists() and (ROOT/'TEST_B_FROZEN.json').exists()
 b=unpack(json.loads((OUT/j['case']/'model_5000.json').read_text()));meta=SYSTEMS[j['index']];family=meta['family'];kind='factor1' if family=='A' else 'memory2' if family=='D' else 'factor2';F,Q,C,R=matrices(meta);oracle=dict(F=F,Q=Q,C=C,R=R,a=0.,P=solve_discrete_lyapunov(F,Q),scale=1.)
 rows=[];reads=[]
 for batch,stage in [('A',1),('B',2)]:
  x,y=generate(j['index'],j['seed'],stage,NTEST,truth=True)
  # This audit knows the true model family. It is not automatic physical discovery.
  if family=='D':y=y[:,:,:1]
  for line in ['direct','recursive','witness','oracle']:
   m=oracle if line=='oracle' else b[line]['models'][kind]
   if m is None:
    rows.append(dict(case=j['case'],seed=j['seed'],condition=meta['id'],batch=batch,line=line,kind=kind,available=False,coordinate=-1));continue
   mu,cov=fitted_post(x,m)
   if family=='D':mu=mu[:,:,:1];cov=cov[:,:1,:1]
   for r in quality(mu,cov,y):rows.append(dict(case=j['case'],seed=j['seed'],condition=meta['id'],batch=batch,line=line,kind=kind,available=True,**r))
 x,y=generate(j['index'],j['seed'],3,20,truth=True)
 if family=='D':y=y[:,:,:1]
 for line in ['direct','recursive','witness','Xonly']:
  if line=='Xonly':features=x[:,:,None];selected='current_X'
  else:
   bank=b[line];selected=min([k for k in KINDS if bank['models'][k] is not None],key=lambda k:float(np.mean(bank['selection_scores'][k])));m=bank['models'][selected];mu,cov=fitted_post(x,m);features=np.concatenate([x[:,:,None],mu],2)
  train=features[:10,200:].reshape(-1,features.shape[-1]);test=features[10:,200:].reshape(-1,features.shape[-1]);yt=y[:10,200:].reshape(-1,y.shape[-1]);yv=y[10:,200:].reshape(-1,y.shape[-1]);train=np.c_[np.ones(len(train)),train];test=np.c_[np.ones(len(test)),test];beta=np.linalg.lstsq(train,yt,rcond=1e-8)[0];pred=test@beta
  for k in range(y.shape[-1]):reads.append(dict(case=j['case'],seed=j['seed'],condition=meta['id'],line=line,selected=selected,coordinate=k,R2=1-np.mean((yv[:,k]-pred[:,k])**2)/np.var(yv[:,k]),corr=np.corrcoef(yv[:,k],pred[:,k])[0,1],MSE=np.mean((yv[:,k]-pred[:,k])**2)))
 print('STATE_AUDIT',j['case'],flush=True);return rows,reads

def main():
 for batch in ['A','B']:
  for f,h in json.loads((ROOT/f'TEST_{batch}_FROZEN.json').read_text())['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 rows=[];reads=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):a,b=f.result();rows.extend(a);reads.extend(b)
 pd.DataFrame(rows).to_csv(OUT/'state_quality.csv',index=False);pd.DataFrame(reads).to_csv(OUT/'state_readout.csv',index=False);(ROOT/'TRUTH_AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),after_prediction_tests=True,oracle_family_diagnostic_only=True,readout_not_physical_posterior=True),indent=2))
if __name__=='__main__':main()
