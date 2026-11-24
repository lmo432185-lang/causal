"""Post-test physical truth audit. No output is returned to fitting or selection."""
import json,gzip,time,numpy as np,pandas as pd
from scipy.special import logsumexp
from scipy.linalg import solve_discrete_lyapunov
from concurrent.futures import ProcessPoolExecutor,as_completed
from settings import *
from producer import generate,matrices,SYSTEMS,digest
from model import unpack
from probability import TIMES,marginal,physical_scores,correlated_alias,nodes
from inference import mix
import native

def qphysical(x,models,prior,view=False):
 ll=[];mu=[];cov=[];full=[]
 for m in models:
  p=native.run(x/m['scale'],m);p['ll']-=np.log(m['scale']);ll.append(p['ll']);mu.append(p['mean'][:,TIMES]*m['scale']);cov.append(p['cov'][TIMES]*m['scale']**2)
  if view:full.append((p['mean'][0,200:400]*m['scale'],p['cov'][200:400]*m['scale']**2))
 lw=np.cumsum(np.stack(ll,-1),1)+np.array(prior);lw-=logsumexp(lw,2)[:,:,None];w=np.exp(lw[:,TIMES]);mu=np.stack(mu,2);cov=np.broadcast_to(np.stack(cov,1)[None],(len(x),len(TIMES),len(models),2,2)).copy()
 v=None
 if view:v=dict(mu=np.stack([a for a,c in full],1),cov=np.stack([c for a,c in full],1),w=np.exp(lw[0,200:400]))
 return mu,cov,w,v

def physical_work(j,batch,b,x,y):
 meta=SYSTEMS[j['index']];base=b['bases']['R'];theta=base['canonical_point'];boots=base['canonical_bootstrap'];priors=json.loads((OUT/'canonical_priors.json').read_text())[j['case']]['R'];rho,wr=nodes();rp,wp=nodes(positive=True);models={
 'INDEP_POINT':([theta],[0.]),'FULLQ_POINT':([correlated_alias(theta,0.)],[0.]),'PARAM_ONLY':(boots,priors),
 'FULLQ_NULL':([correlated_alias(theta,r) for r in rho],np.log(wr)),
 'FULLQ_PARAM':([correlated_alias(m,r) for m in boots for r in rho],np.array([pr+np.log(w) for pr in priors for w in wr])),
 'POSITIVE_PRIOR':([correlated_alias(m,r) for m in boots for r in rp],np.array([pr+np.log(w) for pr in priors for w in wp]))}
 rec=b['models'].get('oldM4_R_factor2')
 if rec is not None:models['RECURSIVE_POINT']=([rec],[0.])
 bayespath=OUT/'bayes'/(j['case']+'.json.gz')
 if bayespath.exists():
  from bayes import make_model
  with gzip.open(bayespath,'rt') as bf:bb=unpack(json.load(bf))
  ms=bb['models8'];lp=np.full(len(ms),-np.log(len(ms)));models['BAYES_POINT']=([bb['point_model']],[0.]);models['BAYES_PARAM']=(ms,lp);models['BAYES_FULL']=([correlated_alias(m,r) for m in ms for r in rho],np.array([pr+np.log(w) for pr in lp for w in wr]))
 F,Q,C,R=matrices(meta);oracle=dict(F=F,Q=Q,C=C,R=R,P=solve_discrete_lyapunov(F,Q),a=0.,scale=1.,kind='factor2');models['ORACLE']=([oracle],[0.]);rows=[];joint=[];example={};target=y[:,TIMES]
 for method in ['INDEP_POINT','FULLQ_POINT','PARAM_ONLY','FULLQ_NULL','FULLQ_PARAM','POSITIVE_PRIOR','RECURSIVE_POINT','ORACLE']+(['BAYES_POINT','BAYES_PARAM','BAYES_FULL'] if bayespath.exists() else []):
  if method not in models:
   for i in range(NTEST):joint.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=method,trajectory=i,available=False,joint_NLL=np.nan,posterior_corr=np.nan,joint_cover50=np.nan,joint_cover80=np.nan,joint_cover90=np.nan,joint_cover95=np.nan))
   continue
  ms,prior=models[method];mu,cov,w,view=qphysical(x,ms,prior,view=j['seed']==4011 and batch=='A' and j['index']>=20);P=len(x)*len(TIMES);mm=mu.reshape(P,len(ms),2);cc=cov.reshape(P,len(ms),2,2);ww=w.reshape(P,len(ms));yy=target.reshape(P,2);scores=physical_scores(mm,cc,ww,yy)
  for k,label in enumerate(['U','V']):
   sc,mean,var=marginal(mm[:,:,k],cc[:,:,k,k],ww,yy[:,k])
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=method,coordinate=label,trajectory=i,available=True,**{name:float(value.reshape(NTEST,len(TIMES))[i].mean()) for name,value in sc.items()}))
  for i in range(NTEST):joint.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=method,trajectory=i,available=True,**{name:float(value.reshape(NTEST,len(TIMES))[i].mean()) for name,value in scores.items() if name.startswith('joint_') or name=='posterior_corr'}))
  if view is not None:
   for key,value in view.items():example[method+'_'+key]=value
 if example:example['truth']=y[0,200:400];example['X']=x[0,200:400];example['times']=np.arange(200,400);np.savez_compressed(OUT/('physical_example_'+j['case']+'.npz'),**example)
 return rows,joint

def signal_work(j,batch,b,x,y):
 Ctrue=matrices(SYSTEMS[j['index']])[2];target=(y@Ctrue)[:,TIMES];rows=[];cache={};done={}
 for name in ['J0','J1','J2','A','B','R0','R2','J1_param','J2_point','MEMORY_POINT']:
  plan=b['plans'][name];ids=plan['ids'];pr=plan.get('prior');key=(tuple(ids),tuple(pr or []))
  if not ids:
   for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,available=False));continue
  if key not in done:
   r,_=mix(x,b['models'],ids,pr,cache);mu=[];var=[]
   for k,p in zip(ids,r['parts']):
    m=b['models'][k];mu.append(np.einsum('i,nti->nt',m['C'],p['post']['mean'][:,TIMES])*m['scale']+m.get('a',0)*x[:,TIMES-1]);var.append(np.einsum('i,tij,j->t',m['C'],p['post']['cov'][TIMES],m['C'])*m['scale']**2)
   mu=np.stack(mu,2);var=np.broadcast_to(np.stack(var,1)[None],mu.shape);w=r['weights'][:,TIMES];sc,mean,v=marginal(mu.reshape(-1,len(ids)),var.reshape(-1,len(ids)),w.reshape(-1,len(ids)),target.ravel());done[key]=sc
  sc=done[key]
  for i in range(NTEST):rows.append(dict(case=j['case'],seed=j['seed'],batch=batch,method=name,trajectory=i,available=True,**{k:float(v.reshape(NTEST,len(TIMES))[i].mean()) for k,v in sc.items()}))
 return rows

def readout_work(j,b):
 x,y=generate(j['index'],j['seed'],13,20,truth=True);rows=[]
 if SYSTEMS[j['index']]['family']=='D':y=y[:,:,:1]
 # A held-out linear readout is not a calibrated physical-state posterior.
 for bank in ['R','D']:
  for route in ['R0','R2','R3']:
   key=b['candidates'][route][bank]['best']
   if key is None:continue
   m=b['models'][key];p=native.run(x/m['scale'],m);f=np.concatenate([x[:,:,None],p['mean']*m['scale']],2);a=f[:10,200:].reshape(-1,f.shape[-1]);z=f[10:,200:].reshape(-1,f.shape[-1]);u=y[:10,200:].reshape(-1,y.shape[-1]);v=y[10:,200:].reshape(-1,y.shape[-1]);a=np.c_[np.ones(len(a)),a];z=np.c_[np.ones(len(z)),z];coef=np.linalg.lstsq(a,u,rcond=1e-8)[0];pred=z@coef
   for k in range(y.shape[-1]):rows.append(dict(case=j['case'],seed=j['seed'],bank=bank,route=route,coordinate=k,R2=1-np.mean((pred[:,k]-v[:,k])**2)/np.var(v[:,k]),MSE=np.mean((pred[:,k]-v[:,k])**2)))
 a=np.c_[np.ones(10*4800),x[:10,200:].reshape(-1)];z=np.c_[np.ones(10*4800),x[10:,200:].reshape(-1)];u=y[:10,200:].reshape(-1,y.shape[-1]);v=y[10:,200:].reshape(-1,y.shape[-1]);pred=z@np.linalg.lstsq(a,u,rcond=1e-8)[0]
 for k in range(y.shape[-1]):rows.append(dict(case=j['case'],seed=j['seed'],bank='none',route='Xonly',coordinate=k,R2=1-np.mean((pred[:,k]-v[:,k])**2)/np.var(v[:,k]),MSE=np.mean((pred[:,k]-v[:,k])**2)))
 return rows

def work(j):
 assert (ROOT/'TEST_B_FROZEN.json').exists()
 with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
 b['plans'].update(json.loads((OUT/'extra_plans.json').read_text())[j['case']]);rows=[];joint=[];signals=[]
 for batch,stage in [('A',11),('B',12)]:
  x,y=generate(j['index'],j['seed'],stage,NTEST,truth=True);record=json.loads((ROOT/f'TEST_{batch}_FROZEN.json').read_text());assert digest(x)==record['input_hashes'][j['case']]
  signals.extend(signal_work(j,batch,b,x,y))
  if SYSTEMS[j['index']]['family'] in ['C','Cgrid']:
   a,c=physical_work(j,batch,b,x,y);rows.extend(a);joint.extend(c)
 read=readout_work(j,b);print('TRUTH',j['case'],flush=True);return rows,joint,signals,read

def main():
 for label in ['TEST_A_FROZEN','TEST_B_FROZEN']:
  for f,h in json.loads((ROOT/(label+'.json')).read_text())['files'].items():
   import hashlib
   assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 results=[[],[],[],[]]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(work,j) for j in json.loads((ROOT/'jobs.json').read_text())]):
   for target,rows in zip(results,f.result()):target.extend(rows)
 for rows,name in zip(results,['state_marginal','state_joint','signal_probability','state_readout']):pd.DataFrame(rows).to_csv(OUT/(name+'.csv.gz'),index=False,compression='gzip')
 (ROOT/'TRUTH_AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),true_states_only_after_tests=True,supervised_calibration=False,state_times=TIMES.tolist(),physical_UV_for_C_only=True,J1_physical_identity='not_applicable_unmatched_coordinates'),indent=2));print('TRUTH AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
