"""X-only switching-AR learning and retrospective event localization.
No true labels, physical coefficients, or event timing enter fit().
"""
import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'; os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
from pathlib import Path
import json,csv,time
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
DT=.05; T=500; SEEDS=[11,22,33]; LAGS=[0,10,20,40,500]
CFG=dict(dt=DT,duration=25,train=200,validation_null=100,test_event=200,test_null=200,test_short=200,training_seeds=SEEDS,noise=[.12,.35,.7],lag_seconds=[0,.5,1,2,'full'],EM_iterations=45,states=2,threshold_null_false_alarm_target=.05,simulation_equation='x[t+1]=x[t]+dt*(1.5-.15*x[t]-1.5*y[t])+sigma*sqrt(dt)*noise',training_model='two-state switching AR: dx=beta*x+mu[state]+Gaussian residual; all parameters and transitions learned from X')
def generate(n,sigma,seed,kind='mixed'):
 rng=np.random.default_rng(seed); y=np.zeros((n,T)); on=np.full(n,-1); off=on.copy(); amp=np.zeros(n)
 for i in range(n):
  if kind=='null' or (kind=='mixed' and rng.random()<.3): continue
  on[i]=rng.integers(120,241); dur=rng.integers(4,13) if kind=='short' else rng.integers(40,101); off[i]=on[i]+dur; amp[i]=rng.uniform(.8,1.2); y[i,on[i]:off[i]]=amp[i]
 x=np.zeros((n,T+1)); x[:,0]=rng.normal(10,.3,n)
 for t in range(T): x[:,t+1]=x[:,t]+DT*(1.5-.15*x[:,t]-1.5*y[:,t])+sigma*np.sqrt(DT)*rng.normal(size=n)
 return dict(x=x,y=y,on=on,off=off,amp=amp)
def emission(x,p):
 d=np.diff(x,axis=1); residual=d[:,:,None]-p['beta']*x[:,:-1,None]-p['mu']; ll=-.5*(residual**2/p['var']+np.log(2*np.pi*p['var'])); scale=ll.max(2); return np.exp(ll-scale[:,:,None]),scale

def fb(x,p):
 e,scale=emission(x,p); n,t,_=e.shape; prior=np.empty_like(e); post=np.empty_like(e); norm=np.empty((n,t)); prior[:,0]=p['pi']
 for j in range(t):
  v=prior[:,j]*e[:,j]; norm[:,j]=v.sum(1); post[:,j]=v/norm[:,j,None]
  if j<t-1: prior[:,j+1]=post[:,j]@p['A']
 b=np.ones_like(e)
 for j in range(t-2,-1,-1):
  z=(e[:,j+1]*b[:,j+1])@p['A'].T; b[:,j]=z/z.sum(1)[:,None]
 g=post*b; g/=g.sum(2)[:,:,None]
 return e,prior,post,b,g,float((np.log(norm)+scale).sum())
def fit(x,seed):
 rng=np.random.default_rng(seed); d=np.diff(x,axis=1); xx=x[:,:-1]; init=np.linalg.lstsq(np.c_[xx.ravel(),np.ones(xx.size)],d.ravel(),rcond=None)[0]; residual=d-init[0]*xx
 p=dict(beta=float(init[0]),mu=np.quantile(residual,[.12,.75])+rng.normal(0,.002,2),var=float(np.var(residual)),A=np.array([[.96,.04],[.01,.99]]),pi=np.array([.15,.85]))
 trace=[]
 for k in range(CFG['EM_iterations']):
  e,prior,post,b,g,ll=fb(x,p); trace.append(ll)
  counts=np.zeros((2,2))
  for j in range(T-1):
   v=post[:,j,:,None]*p['A'][None,:,:]*(e[:,j+1]*b[:,j+1])[:,None,:]; v/=v.sum((1,2))[:,None,None]; counts+=v.sum(0)
  # Small symmetric pseudocount prevents exact absorbing states.
  counts+=.01; A=counts/counts.sum(1)[:,None]; pi=g[:,0].mean(0); pi=np.maximum(pi,1e-6); pi/=pi.sum()
  mass=g.sum((0,1)); sx=(g*xx[:,:,None]).sum((0,1)); sy=(g*d[:,:,None]).sum((0,1)); sxx=(xx*xx).sum(); sxy=(xx*d).sum()
  mat=np.array([[sxx,sx[0],sx[1]],[sx[0],mass[0],0],[sx[1],0,mass[1]]]); rhs=np.array([sxy,sy[0],sy[1]]); co=np.linalg.solve(mat+np.eye(3)*1e-9,rhs)
  var=float((g*(d[:,:,None]-co[0]*xx[:,:,None]-co[1:])**2).sum()/d.size)
  p=dict(beta=float(co[0]),mu=co[1:],var=max(var,1e-8),A=A,pi=pi)
 order=np.argsort(p['mu']); p['mu']=p['mu'][order]; p['A']=p['A'][order][:,order]; p['pi']=p['pi'][order]
 return p,trace

def infer(x,p):
 e,prior,post,b,g,ll=fb(x,p); n,t,_=e.shape; out={0:prior[:,:,0],500:g[:,:,0]}
 # z_t inferred using X up through t+lag; e_t is first seen at t+1.
 for lag in [10,20,40]:
  beta=np.ones_like(e)
  for depth in range(lag):
   new=np.ones_like(e); new[:,-1]=e[:,-1]
   new[:,:-1]=e[:,:-1]*(beta[:,1:]@p['A'].T)
   beta=new/np.maximum(new.sum(2)[:,:,None],1e-300)
  pp=prior*beta; pp/=pp.sum(2)[:,:,None]; out[lag]=pp[:,:,0]
 return out,(e,prior,post,b,g)

def smooth_to_endpoint(e,prior,p,endpoint):
 # Endpoint is an X sample index; observations include increments [0,endpoint).
 end=min(endpoint,T); n=e.shape[0]; beta=np.ones((n,2)); probs=np.empty((n,end))
 for j in range(end-1,-1,-1):
  beta=e[:,j]*beta; beta/=beta.sum(1)[:,None]; z=prior[:,j]*beta; z/=z.sum(1)[:,None]; probs[:,j]=z[:,0]
  beta=beta@p['A'].T; beta/=beta.sum(1)[:,None]
 return probs

def first_run(mask,length=2):
 if len(mask)<length: return None
 ix=np.flatnonzero(np.convolve(mask.astype(int),np.ones(length,dtype=int),'valid')==length)
 return int(ix[0]) if len(ix) else None

def stats(v):
 a=np.asarray(v,float); a=a[np.isfinite(a)]; n=len(a)
 if not n:return dict(n=0,mean=None,median=None,ci95=None)
 se=a.std(ddof=1)/np.sqrt(n) if n>1 else 0
 return dict(n=n,mean=float(a.mean()),median=float(np.median(a)),ci95=[float(a.mean()-1.972*se),float(a.mean()+1.972*se)])
def score(case,label,out,threshold,seed,sigma):
 rows=[]
 for lag,pp in out.items():
  for i in range(len(pp)):
   on=int(case['on'][i]); off=int(case['off'][i]); true=case['y'][i]>0; active=pp[i]>=threshold
   start=first_run(active); detected=start is not None; row=dict(sigma=sigma,seed=seed,case=label,trajectory=i,lag=lag,threshold=float(threshold),detected=int(detected),onset_error=np.nan,early_alarm=0,miss=0,brier=float(np.mean((pp[i]-true)**2)),state_mae=float(np.mean(np.abs(pp[i]-true))),onset_signed=np.nan)
   if on>=0:
    # No matching window selected using truth: first inferred sustained episode wins.
    row['miss']=int(not detected); row['early_alarm']=int(detected and start<on-10)
    if detected: row['onset_error']=abs(start-on)*DT; row['onset_signed']=(start-on)*DT
   rows.append(row)
 return rows

def main():
 (OUT/'config.json').write_text(json.dumps(CFG,indent=2)); rows=[]; snap=[]; checks={}; examples={}; starttime=time.time()
 for gi,sigma in enumerate(CFG['noise']):
  train=generate(200,sigma,1000+gi); val=generate(100,sigma,2000+gi,'null'); cases={k:generate(200,sigma,3000+gi*10+j,k) for j,k in enumerate(['event','null','short'])}
  for seed in SEEDS:
   p,trace=fit(train['x'],seed); vo,_=infer(val['x'],p)
   # One common threshold for all lookahead methods: calibrated on the maximum
   # across time AND methods on each null validation trajectory (conservative).
   maxima=np.max(np.stack([v.max(1) for v in vo.values()]),0); threshold=float(np.quantile(maxima,.95,method='higher'))
   # Keep the full posterior probabilities: threshold is NOT fitted with Y.
   checks[f'{sigma}_{seed}']=dict(likelihood_increases=bool(np.min(np.diff(trace))>=-1e-4),min_ll_step=float(np.min(np.diff(trace))),threshold=threshold,fit={k:np.asarray(v).tolist() for k,v in p.items()})
   np.savez(OUT/f'model_{gi}_{seed}.npz',**p,likelihood=trace)
   for label,case in cases.items():
    out,detail=infer(case['x'],p); rows.extend(score(case,label,out,threshold,seed,sigma)); e,prior,post,b,g=detail
    if seed==11 and label=='event': examples[f'{sigma}']=dict(x=case['x'][0],y=case['y'][0],on=case['on'][0],off=case['off'][0],threshold=threshold,**{f'p{k}':v[0] for k,v in out.items()})
    # Real-time alert at X endpoint j+1 uses filtered posterior z_j.
    if label=='event':
     for i in range(200):
      alert=first_run(post[i,:,0]>=threshold); on=int(case['on'][i]); off=int(case['off'][i])
      # two-step persistence: event first run j triggers at X index j+2.
      endpoint=alert+2 if alert is not None else None
      for additional in [0,10,20,40]:
       error=np.nan; estimate=None
       if endpoint is not None:
        pp=smooth_to_endpoint(e[i:i+1],prior[i:i+1],p,min(T,endpoint+additional))[0]; estimate=first_run(pp>=threshold)
        if estimate is not None:error=abs(estimate-on)*DT
       snap.append(dict(sigma=sigma,seed=seed,trajectory=i,additional_seconds=additional*DT,alert_delay=np.nan if endpoint is None else (endpoint-on)*DT,alert_early=int(endpoint is not None and endpoint<on),alert_during=int(endpoint is not None and on<=endpoint<=off),alert_miss=int(endpoint is None),backdated_error=error,estimate_available=int(estimate is not None)))
   print('noise',sigma,'seed',seed,'done',round(time.time()-starttime,1),'threshold',threshold,flush=True)
 with (OUT/'per_trajectory.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (OUT/'snapshots.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(snap[0]));w.writeheader();w.writerows(snap)
 (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
 np.savez_compressed(OUT/'examples.npz',**{g+'_'+k:v for g,d in examples.items() for k,v in d.items()})
 summarize(rows,snap,checks)

def summarize(rows,snap,checks):
 result={}; changes={}
 for sigma in CFG['noise']:
  result[str(sigma)]={}
  for case in ['event','null','short']:
   result[str(sigma)][case]={}
   for lag in LAGS:
    sub=[r for r in rows if r['sigma']==sigma and r['case']==case and r['lag']==lag]; summ={}
    for key in ['detected','onset_error','onset_signed','early_alarm','miss','brier','state_mae']:
     vals=[np.nanmean([r[key] for r in sub if r['trajectory']==j]) if any(np.isfinite(r[key]) for r in sub if r['trajectory']==j) else np.nan for j in range(200)]; summ[key]=stats(vals)
    summ['seed_onset_MAE']={str(seed):stats([r['onset_error'] for r in sub if r['seed']==seed])['mean'] for seed in SEEDS}; result[str(sigma)][case][str(lag)]=summ
  result[str(sigma)]['snapshot']={}
  for future in [0,.5,1,2]:
   sub=[r for r in snap if r['sigma']==sigma and r['additional_seconds']==future]; ss={}
   for key in ['alert_delay','alert_early','alert_during','alert_miss','backdated_error','estimate_available']:
    vals=[np.nanmean([r[key] for r in sub if r['trajectory']==j]) if any(np.isfinite(r[key]) for r in sub if r['trajectory']==j) else np.nan for j in range(200)]; ss[key]=stats(vals)
   result[str(sigma)]['snapshot'][str(future)]=ss
  # Paired Brier improvement uses all test trajectories (no detected-only selection).
  sub=[r for r in rows if r['sigma']==sigma and r['case']=='event']; diff=[]
  for j in range(200):
   diff.append(np.mean([r['brier'] for r in sub if r['trajectory']==j and r['lag']==0])-np.mean([r['brier'] for r in sub if r['trajectory']==j and r['lag']==20]))
  changes[str(sigma)]=stats(diff)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)); (OUT/'paired_improvement.json').write_text(json.dumps(changes,indent=2))
 print(json.dumps({g:{case:{lag:{k:v[k]['mean'] for k in ['onset_error','miss','brier']} for lag,v in d[case].items()} for case in ['event','short']} for g,d in result.items()},indent=2),flush=True)
if __name__=='__main__':main()
