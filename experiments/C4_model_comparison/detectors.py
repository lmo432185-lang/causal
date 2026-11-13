import numpy as np
from config import *
from source_C0 import templates
from legacy_stream import hmm_score,reference_scores,alarms,calibrate

def cusum_components(x,base):
 r=-(x[:,1:]-base['q']*x[:,:-1]-base['c'])/np.sqrt(base['var']);a=np.zeros((len(x),len(CUSUM_ALLOWANCES)));out=np.zeros((len(x),len(CUSUM_ALLOWANCES),r.shape[1]//BLOCK))
 for t in range(r.shape[1]):
  a=np.maximum(0,a+r[:,t,None]-np.array(CUSUM_ALLOWANCES)[None,:])
  if (t+1)%BLOCK==0:out[:,:,(t+1)//BLOCK-1]=a
 return out

def structured(x,base,fixed):
 n,T=x.shape[0],x.shape[1]-1;U=T//BLOCK;starts=np.arange(10,T,BLOCK);K=len(ALLTAU);S=len(starts);a=-np.log(base['q'])/DT;q=base['q'];kk=(1-q)/a;j=(np.exp(-DT/ALLTAU)-q)/(a-1/ALLTAU);power=np.exp(-np.arange(BLOCK)[None,:]*DT/ALLTAU[:,None]);norm=np.cumsum(templates(T,base,ALLTAU)**2,axis=1);weighted=np.zeros((n,K,S))
 names=['M2','M3']+ORACLES;out={name:np.zeros((n,U)) for name in names}
 for name in names:
  for field in ['on','tau','amp']:out[f'{field}_{name}']=np.full((n,U),np.nan)
 residual=-(x[:,1:]-q*x[:,:-1]-base['c']);cs=np.c_[np.zeros(n),np.cumsum(residual,axis=1)];ii=np.arange(n)
 for u in range(U):
  begin=u*BLOCK;end=begin+BLOCK;m=np.searchsorted(starts,begin,side='right')
  if m==0:continue
  st=starts[:m];br=residual[:,begin:end]@power.T;decay=np.exp(-(begin-st)[None,:]*DT/ALLTAU[:,None]);weighted[:,:,:m]+=br[:,:,None]*decay[None,:,:];dot=kk*(cs[:,end,None]-cs[:,st])[:,None,:]-j[None,:,None]*weighted[:,:,:m];np.maximum(dot,0,out=dot);dot*=dot;dot/=base['var']*norm[:,end-st-1][None,:,:]
  for name,ki0 in [('M2',fixed),('M3',None)]+[(name,41+i) for i,name in enumerate(ORACLES)]:
   if ki0 is None:best=dot[:,:41,:].reshape(n,-1).argmax(1);ki,si=np.divmod(best,m)
   else:ki=np.full(n,ki0);si=dot[:,ki0,:].argmax(1)
   value=dot[ii,ki,si];out[name][:,u]=value;out[f'on_{name}'][:,u]=st[si]*DT;out[f'tau_{name}'][:,u]=ALLTAU[ki];out[f'amp_{name}'][:,u]=np.sqrt(value*base['var']/norm[ki,end-st[si]-1])
 return out

def score_batch(x,base,p,fixed,scales):
 out=structured(x,base,fixed);out['M1']=hmm_score(x,p);ref=reference_scores(x,base);out['X']=ref['M4_X'];out['FAIL']=ref['M4_E'];cu=cusum_components(x,base)/np.array(scales)[None,:,None];out['CUSUM']=cu.max(1);out['CUSUM_choice']=np.array(CUSUM_ALLOWANCES)[cu.argmax(1)]
 return out

def all_scores(x,base,p,fixed,scales,batch=500):
 parts=[]
 for i in range(0,len(x),batch):parts.append(score_batch(x[i:i+batch],base,p,fixed,scales))
 return {key:np.concatenate([v[key] for v in parts],axis=0) for key in parts[0]}

def local_forecast(x,base,origin,window,horizon=2):
 end=round(origin/DT);width=round(window/DT);r=x[:,1:end+1]-base['q']*x[:,:end]-base['c'];shift=np.minimum(r[:,-width:].mean(1),0);h=round(horizon/DT);qh=base['q']**h;geom=(1-qh)/(1-base['q']);return qh*x[:,end]+(base['c']+shift)*geom

def forecast_from_scores(x,sc,base,p,tau,origin):
 end=round(origin/DT);u=round(origin/CLOCK)-1;h=round(2/DT);qh=base['q']**h;geom=(1-qh)/(1-base['q']);a=-np.log(base['q'])/DT;out={}
 for model,name in [('M2','M2'),('M3','M3'),('M2_oracle',f'oracle_{tau}')]:
  tt=sc[f'tau_{name}'][:,u];on=sc[f'on_{name}'][:,u];amp=sc[f'amp_{name}'][:,u];age=np.maximum(origin-on,0);jh=(np.exp(-2/tt)-qh)/(a-1/tt);kh=(1-qh)/a;out[model]=qh*x[:,end]+base['c']*geom-amp*(kh-np.exp(-age/tt)*jh)
 post=np.c_[sc['M1'][:,u],1-sc['M1'][:,u]];xx=x[:,end].copy()
 for _ in range(h):post=post@p['A'];xx=(1+p['beta'])*xx+post@p['mu']
 out['M1']=xx
 return out
