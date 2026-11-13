import numpy as np
from config import *
from legacy_stream import structured_scores,reference_scores,calibrate,alarms
from source_C0 import templates

def ramp_integral(age,R,base):
 # Exact integral int_0^dt exp(-a(dt-s))*clip((age+s)/R,0,1) ds.
 a=-np.log(base['q'])/DT;q=base['q'];k=(1-q)/a;ell=(DT-k)/a;age=np.asarray(age);R=np.asarray(R);p=np.clip(R-age,0,DT);v=np.exp(-a*(DT-p));A0=(v-q)/a;A1=v*(p/a-1/a**2)+q/a**2;cross=(age*A0+A1)/R+(1-v)/a
 return np.where(age>=R,k,np.where(age+DT<=R,(age*k+ell)/R,cross))

def generic_scores(x,base):
 n,TT=len(x),x.shape[1]-1;U=TT//BLOCK;starts=np.arange(10,TT,BLOCK);R=2*GRID;L=np.floor(R/DT).astype(int);q=base['q'];a=-np.log(q)/DT;k=(1-q)/a;ell=(DT-k)/a;A=DT*k/R;B=ell/R;hcross=ramp_integral(L*DT,R,base);h=ramp_integral(np.arange(TT)[None,:]*DT,R[:,None],base);norm=np.cumsum(h*h,axis=1)
 r=-(x[:,1:]-q*x[:,:-1]-base['c']);cs=np.c_[np.zeros(n),np.cumsum(r,axis=1)];ct=np.c_[np.zeros(n),np.cumsum(r*np.arange(TT)[None,:],axis=1)];out={m:np.zeros((n,U)) for m in ['Direct','GLR_Ramp']}
 for m in out.copy():
  for field in ['on','amp','shape']:out[field+'_'+m]=np.full((n,U),np.nan)
 ii=np.arange(n)
 for u in range(U):
  end=(u+1)*BLOCK;st=starts[starts<=end-BLOCK]
  if len(st)==0:continue
  plain=cs[:,end,None]-cs[:,st];age_count=end-st;direct=np.maximum(plain,0)**2/(base['var']*age_count[None,:]);best=direct.argmax(1);out['Direct'][:,u]=direct[ii,best];out['on_Direct'][:,u]=st[best]*DT;out['amp_Direct'][:,u]=np.maximum(plain[ii,best],0)/(age_count[best]*k);out['shape_Direct'][:,u]=0
  er=np.minimum(end,st[None,:]+L[:,None]);sr=cs[:,er]-cs[:,st][:,None,:];tr=ct[:,er]-ct[:,st][:,None,:];dot=A[None,:,None]*(tr-st[None,None,:]*sr)+B[None,:,None]*sr;cross_index=st[None,:]+L[:,None];dot+=r[:,np.minimum(cross_index,end-1)]*hcross[None,:,None]*(cross_index<end)[None,:,:];ep=np.minimum(end,cross_index+1);dot+=k*(cs[:,end,None,None]-cs[:,ep]);np.maximum(dot,0,out=dot);dot*=dot;dot/=base['var']*norm[:,end-st-1][None,:,:];flat=dot.reshape(n,-1).argmax(1);ki,si=np.divmod(flat,len(st));value=dot[ii,ki,si];out['GLR_Ramp'][:,u]=value;out['on_GLR_Ramp'][:,u]=st[si]*DT;out['amp_GLR_Ramp'][:,u]=np.sqrt(value*base['var']/norm[ki,end-st[si]-1]);out['shape_GLR_Ramp'][:,u]=GRID[ki]
 return out

def cusum(x,base,scales):
 rr=-(x[:,1:]-base['q']*x[:,:-1]-base['c'])/np.sqrt(base['var']);allow=np.array([.1,.25,.5,1,2]);acc=np.zeros((len(x),5));out=np.zeros((len(x),rr.shape[1]//BLOCK))
 for t in range(rr.shape[1]):
  acc=np.maximum(0,acc+rr[:,t,None]-allow[None,:])
  if (t+1)%BLOCK==0:out[:,(t+1)//BLOCK-1]=(acc/np.array(scales)[None,:]).max(1)
 return out

def score_batch(x,base,fixed,scales):
 old=structured_scores(x,base,fixed);out={'M3':old['M3'],'M2':old['M2'],'on_M3':old['onset_M3'],'on_M2':old['onset_M2'],'shape_M3':old['tau_M3'],'shape_M2':np.full_like(old['M2'],GRID[fixed])};norm=np.cumsum(templates(x.shape[1]-1,base,GRID)**2,axis=1)
 for model in ['M3','M2']:
  amp=np.full_like(out[model],np.nan)
  for u in range(amp.shape[1]):
   valid=np.isfinite(out['on_'+model][:,u]);on=np.nan_to_num(out['on_'+model][:,u]);ki=np.searchsorted(GRID,np.nan_to_num(out['shape_'+model][:,u],nan=GRID[0]));ki=np.clip(ki,0,40);age=(u+1)*BLOCK-np.rint(on/DT).astype(int);amp[valid,u]=np.sqrt(out[model][valid,u]*base['var']/norm[ki[valid],age[valid]-1])
  out['amp_'+model]=amp
 # Independent reduced-form likelihood equality is checked analytically/brute-force.
 # Reuse its identical optimized sufficient statistics, avoiding redundant work.
 out['Ablation']=out['M3'];out['on_Ablation']=out['on_M3'];out['amp_Ablation']=out['amp_M3'];out['shape_Ablation']=out['shape_M3'];out.update(generic_scores(x,base));out['CUSUM']=cusum(x,base,scales);ref=reference_scores(x,base);out['FAIL']=ref['M4_E'];out['X']=ref['M4_X'];return out

def scores(x,base,fixed,scales,batch=400):
 chunks=[score_batch(x[i:i+batch],base,fixed,scales) for i in range(0,len(x),batch)];return {k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}

def predict(x,sc,base,origin,drift_window):
 end=round(origin/DT);u=round(origin/CLOCK)-1;steps=40;q=base['q'];a=-np.log(q)/DT;qh=q**steps;geom=(1-qh)/(1-q);pred={}
 for model in ['M3','M2','Ablation']:
  tau=sc['shape_'+model][:,u];on=sc['on_'+model][:,u];amp=sc['amp_'+model][:,u];age=np.maximum(origin-on,0);J=(np.exp(-2/tau)-qh)/(a-1/tau);pred[model]=qh*x[:,end]+base['c']*geom-amp*((1-qh)/a-np.exp(-age/tau)*J)
 amp=sc['amp_Direct'][:,u];pred['Direct']=qh*x[:,end]+base['c']*geom-amp*(1-qh)/a
 xx=x[:,end].copy();amp=sc['amp_GLR_Ramp'][:,u];R=2*sc['shape_GLR_Ramp'][:,u];on=sc['on_GLR_Ramp'][:,u]
 for t in range(steps):xx=q*xx+base['c']-amp*ramp_integral(origin+t*DT-on,R,base)
 pred['GLR_Ramp']=xx
 r=x[:,1:end+1]-q*x[:,:end]-base['c'];shift=np.minimum(r[:,-round(drift_window/DT):].mean(1),0);pred['CUSUM']=qh*x[:,end]+(base['c']+shift)*geom;return pred
