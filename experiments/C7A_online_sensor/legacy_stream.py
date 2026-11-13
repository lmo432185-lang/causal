"""Strictly causal monitoring. No hidden-truth arguments in detector functions."""
import numpy as np
from scipy.special import logsumexp
from source_C0 import TAUGRID,templates
DT=.05;BLOCK=5;CLOCK=DT*BLOCK
METHODS=['M1','M2','M3','M4_X','M4_E']

def structured_scores(x,base,fixed):
 n,T=x.shape[0],x.shape[1]-1;updates=T//BLOCK;starts=np.arange(10,T,BLOCK);K=len(TAUGRID);S=len(starts);a=-np.log(base['q'])/DT;q=base['q'];k=(1-q)/a;j=(np.exp(-DT/TAUGRID)-q)/(a-1/TAUGRID);power=np.exp(-np.arange(BLOCK)[None,:]*DT/TAUGRID[:,None]);norm=np.cumsum(templates(T,base)**2,axis=1)
 weighted=np.zeros((n,K,S));s2=np.zeros((n,updates));s3=np.zeros_like(s2);on2=np.full_like(s2,np.nan);on3=np.full_like(s2,np.nan);tau3=np.full_like(s2,np.nan)
 # Residual at index t is observed only when X_(t+1) arrives.
 residual=-(x[:,1:]-q*x[:,:-1]-base['c']);cumulative=np.c_[np.zeros(n),np.cumsum(residual,axis=1)]
 for u in range(updates):
  begin=u*BLOCK;end=begin+BLOCK;m=np.searchsorted(starts,begin,side='right')
  if m==0:continue
  current_starts=starts[:m];block_r=residual[:,begin:end]@power.T;decay=np.exp(-(begin-current_starts)[None,:]*DT/TAUGRID[:,None]);weighted[:,:,:m]+=block_r[:,:,None]*decay[None,:,:]
  plain=cumulative[:,end,None]-cumulative[:,current_starts];dot=k*plain[:,None,:]-j[None,:,None]*weighted[:,:,:m];np.maximum(dot,0,out=dot);dot*=dot;dot/=base['var']*norm[:,end-current_starts-1][None,:,:]
  idx2=np.argmax(dot[:,fixed,:],axis=1);s2[:,u]=dot[np.arange(n),fixed,idx2];on2[:,u]=current_starts[idx2]*DT
  best=dot.reshape(n,-1).argmax(1);ki,si=np.divmod(best,m);s3[:,u]=dot[np.arange(n),ki,si];on3[:,u]=current_starts[si]*DT;tau3[:,u]=TAUGRID[ki]
 return dict(M2=s2,M3=s3,onset_M2=on2,onset_M3=on3,tau_M3=tau3)

def hmm_score(x,p):
 n,T=x.shape[0],x.shape[1]-1;prior=np.broadcast_to(p['pi'],(n,2)).copy();score=np.empty((n,T//BLOCK))
 for t in range(T):
  residual=(x[:,t+1]-x[:,t])[:,None]-p['beta']*x[:,t,None]-p['mu'];ll=-.5*(residual**2/p['var']+np.log(2*np.pi*p['var']));lp=np.log(np.maximum(prior,1e-300))+ll;lp-=logsumexp(lp,axis=1)[:,None];post=np.exp(lp)
  if (t+1)%BLOCK==0:score[:,(t+1)//BLOCK-1]=post[:,0]
  prior=post@p['A']
 return score

def reference_scores(x,base,horizon_steps=20,window=4):
 n,T=x.shape[0],x.shape[1]-1;u=T//BLOCK;score_x=np.empty((n,u));score_e=np.zeros((n,u));mean=x[:,0].copy();variance=0.;qh=base['q']**horizon_steps;ch=base['c']*(1-qh)/(1-base['q']);vh=base['var']*(1-qh*qh)/(1-base['q']**2);errors=[]
 for t in range(1,T+1):
  mean=base['q']*mean+base['c'];variance=base['q']**2*variance+base['var']
  if t%BLOCK:continue
  j=t//BLOCK-1;score_x[:,j]=(mean-x[:,t])/np.sqrt(variance)
  if t>=horizon_steps:
   # This prediction was emitted at t-horizon_steps. It matures only now.
   error=(x[:,t]-(qh*x[:,t-horizon_steps]+ch))**2/vh;errors.append(error)
   if len(errors)>=window:score_e[:,j]=np.mean(np.stack(errors[-window:]),axis=0)
 return dict(M4_X=score_x,M4_E=score_e)

def scores(x,base,p,fixed):
 result=structured_scores(x,base,fixed);result['M1']=hmm_score(x,p);result.update(reference_scores(x,base));return result

def calibrate(score,k,target=.05):
 mins=np.minimum.reduce([score[:,offset:score.shape[1]-k+offset+1] for offset in range(k)]);maxima=mins.max(1);return float(np.quantile(maxima,1-target,method='higher'))
def alarms(score,threshold,k):
 mask=score>threshold;windows=np.logical_and.reduce([mask[:,offset:mask.shape[1]-k+offset+1] for offset in range(k)]);exists=windows.any(1);idx=windows.argmax(1)+k-1;return np.where(exists,(idx+1)*CLOCK,np.nan),np.where(exists,idx,-1)
def episodes(score,threshold,k=2,hysteresis=False):
 found=[]
 for values in score:
  times=[];armed=True;pending=0;below=0;low=.5*threshold if hysteresis else threshold
  for j,value in enumerate(values):
   if armed:
    pending=pending+1 if value>threshold else 0
    if pending>=k:times.append((j+1)*CLOCK);armed=False;pending=0;below=0
   else:
    below=below+1 if value<low else 0
    if below>=2:armed=True;pending=0;below=0
  found.append(times)
 return found
