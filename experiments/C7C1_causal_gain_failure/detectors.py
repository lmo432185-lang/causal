import numpy as np
from scipy.signal import lfilter
from config import *
from physics import forcing
from legacy_stream import structured_scores

def nominal(x,I,model):
 q=model['q'];c=model['c'];a=-np.log(q)/DT;k=(1-q)/a;L=(DT-k)/a;h,m,_,_=forcing(I,model['tau'],q);h=model['f0']*k+model['gain']*h;m=model['f0']*L+model['gain']*m;e=x[:,1:]-q*x[:,:-1]-c+h;return e,h,m

def cusum_bank(e,var):
 acc=np.zeros((len(e),len(ALLOW)));out=np.zeros((len(e),e.shape[1]//BLOCK,len(ALLOW)))
 for t in range(e.shape[1]):
  acc=np.maximum(0,acc+e[:,t,None]/np.sqrt(var)-ALLOW[None,:])
  if (t+1)%BLOCK==0:out[:,(t+1)//BLOCK-1,:]=acc
 return out

def forecast_score(x,h,model):
 q=model['q'];var=model['var'];n,steps=h.shape;drift=model['c']-h;D=np.c_[np.zeros(n),lfilter([1],[1,-q],drift,axis=1)];H=20;qh=q**H;vh=var*(1-qh*qh)/(1-q*q);out=np.zeros((n,steps//BLOCK));errors=[]
 for end in range(BLOCK,steps+1,BLOCK):
  if end<H:continue
  pred=qh*x[:,end-H]+D[:,end]-qh*D[:,end-H];err=(x[:,end]-pred)**2/vh;errors.append(err)
  if len(errors)>=4:out[:,end//BLOCK-1]=np.mean(np.stack(errors[-4:]),axis=0)
 return out

def scores(x,I,model,scales):
 n=len(x);steps=x.shape[1]-1;updates=steps//BLOCK;e,h,moment=nominal(x,I,model);var=model['var'];q=model['q'];a=-np.log(q)/DT;k=(1-q)/a;L=(DT-k)/a
 origins=np.arange(.5,steps*DT,CLOCK);age=np.arange(steps)[None,None,:]*DT-origins[None,:,None];dur=DURATIONS[:,None,None];safe=np.where(dur>0,dur,1);F=np.where(dur==0,(age>=-1e-10).astype(float),np.clip(age/safe,0,1));slope=np.where((dur>0)&(age>=-1e-10)&(age<dur-1e-10),1/safe,0);generic=k*F+L*slope;normD=np.cumsum(generic*generic,axis=2);dotD=np.zeros((n,len(DURATIONS),len(origins)));dotG=np.zeros_like(dotD);normG=np.zeros_like(dotD)
 out={name:np.zeros((n,updates)) for name in ['GLR','GainGLR']};meta={name:np.full((n,updates),np.nan) for name in ['rho','onset','duration','beta_estimate']}
 for u in range(updates):
  begin=u*BLOCK;end=begin+BLOCK;count=int(np.sum(origins<=begin*DT+1e-9))
  if count==0:continue
  gg=generic[:,:count,begin:end];dotD[:,:,:count]+=np.einsum('nt,dst->nds',e[:,begin:end],gg,optimize=True);nd=normD[:,:count,end-1];v=np.maximum(dotD[:,:,:count],0)**2/(var*np.maximum(nd[None,:,:],1e-300));out['GLR'][:,u]=v.reshape(n,-1).max(1)
  lg=h[:,None,None,begin:end]*F[None,:,:count,begin:end]+moment[:,None,None,begin:end]*slope[None,:,:count,begin:end];dotG[:,:,:count]+=np.sum(e[:,None,None,begin:end]*lg,axis=3);normG[:,:,:count]+=np.sum(lg*lg,axis=3);ng=normG[:,:,:count];dg=dotG[:,:,:count];rho=np.clip(dg/np.maximum(ng,1e-300),0,1);value=(2*rho*dg-rho*rho*ng)/var;flat=value.reshape(n,-1).argmax(1);di,si=np.divmod(flat,count);out['GainGLR'][:,u]=value[np.arange(n),di,si];rr=rho[np.arange(n),di,si];on=origins[si];dd=DURATIONS[di];now=end*DT;fraction=np.where(dd==0,1,np.clip((now-on)/np.where(dd>0,dd,1),0,1));meta['rho'][:,u]=rr;meta['onset'][:,u]=on;meta['duration'][:,u]=dd;meta['beta_estimate'][:,u]=1-rr*fraction
 out['GainSpecific']=np.where(out['GainGLR']-out['GLR']>ATTR_MARGIN,out['GainGLR'],0)
 out['CUSUM']=(cusum_bank(e,var)/np.asarray(scales)[None,None,:]).max(2)
 # Frozen M3's sign and healthy-command adapter only. Its score implementation is unchanged.
 aux=np.c_[np.zeros(n),lfilter([-1],[1,-q],e,axis=1)];out['M3_adapted']=structured_scores(aux,dict(q=q,c=0.,var=var),0)['M3'];out['PredictionError']=forecast_score(x,h,model)
 return out,meta

def alarm(sc,threshold):
 hit=(sc[:,1:]>threshold)&(sc[:,:-1]>threshold);has=hit.any(1);idx=hit.argmax(1)+1;return np.where(has,(idx+1)*CLOCK,np.nan),np.where(has,idx,-1)

def maxima(sc):return np.minimum(sc[:,1:],sc[:,:-1]).max(1)
def calibrate(v,target):return float(np.quantile(v,1-target,method='higher'))
