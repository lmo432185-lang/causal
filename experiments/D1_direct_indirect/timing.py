"""Conditional known-family timing diagnostic; parameters learned from normal X only."""
import json
import numpy as np,pandas as pd
from scipy.linalg import expm,solve_discrete_are
from scipy.signal import ss2tf,lfilter
from config import *
from physics import system,simulate
from spectral import library,periodogram
from states import fitted_system

DURATION=32.;NT=1280;BLOCK=10;H=40

def batch_rec(A,B,C,x):
 # x shapeN,T,input; z_t=A zprev+B x_t, outputCz_t
 out=np.zeros((x.shape[0],x.shape[1],C.shape[0]));D=C@B
 for j in range(B.shape[1]):
  nums,den=ss2tf(A,B,C@A,D,input=j)
  for k,num in enumerate(nums):out[:,:,k]+=lfilter(np.real_if_close(num),np.real_if_close(den),x[:,:,j],axis=1)
 return out

def estimate(x,theta):
 _,F,Q,P=fitted_system(2,theta);d=3;HH=np.array([[0.,0.,1.]]);Pp=solve_discrete_are(F.T,HH.T,Q,np.zeros((1,1)));K=Pp@HH.T/(HH@Pp@HH.T).item();Af=(np.eye(d)-K@HH)@F;mf=batch_rec(Af,K,np.eye(d),x[:,:,None]);Pf=(np.eye(d)-K@HH)@Pp;J=Pf@F.T@np.linalg.inv(Pp);v=np.einsum('ij,ntj->nti',np.eye(d)-J@F,mf)[:,::-1,:].copy();v[:,0]=mf[:,-1];ms=batch_rec(J,np.eye(d),np.eye(d),v)[:,::-1,:]
 predprev=np.einsum('ij,ntj->nti',F,np.concatenate([np.zeros((len(x),1,d)),mf[:,:-1]],axis=1));innov=(x-predprev[:,:,-1])/np.sqrt(Pp[-1,-1]);accp=np.zeros(len(x));accm=np.zeros(len(x));cusum=[]
 for t in range(x.shape[1]):
  accp=np.maximum(0,accp+innov[:,t]-.1);accm=np.maximum(0,accm-innov[:,t]-.1)
  if t>0 and t%BLOCK==0:cusum.append(np.maximum(accp,accm).copy())
 Fh=np.linalg.matrix_power(F,H);Vh=P-Fh@(P-Pf)@Fh.T;errors=np.full_like(x,np.nan);mu=mf[:,:-H]@Fh[-1];errors[:,H:]=(x[:,H:]-mu)**2/Vh[-1,-1];scerr=np.zeros((len(x),NT//BLOCK));recent=[]
 for end in range(BLOCK,NT+1,BLOCK):
  if end<H:continue
  recent.append(errors[:,end]);
  if len(recent)>=4:scerr[:,end//BLOCK-1]=np.mean(np.stack(recent[-4:]),axis=0)
 return dict(SourceY_live=abs(mf[:,BLOCK::BLOCK,0])/np.sqrt(P[0,0]),InnovationCUSUM=np.stack(cusum,1),PredictionError=scerr,SourceY_retrospective=abs(ms[:,BLOCK::BLOCK,0])/np.sqrt(P[0,0])),errors,mf,ms

def data(n,sigma,seed,mode):
 params=dict(kind='indirect',a=.25,c=2.,alpha=1.,beta=1.,sigma=sigma);A,F,Q,P=system(**params);Af,Ff,Qf,Pf=system('indirect',.25,2.,alpha=1.,beta=0.,sigma=sigma);rng=np.random.default_rng(seed);state=rng.normal(size=(n,3))@np.linalg.cholesky(P).T;noise=rng.normal(size=(n,NT,3));L=np.linalg.cholesky(Q);Lf=np.linalg.cholesky(Qf+np.eye(3)*1e-15);on=np.random.default_rng(np.random.SeedSequence([seed,47])).integers(40,73,n)*.25;out=np.empty((n,NT+1,3));out[:,0]=state;Bmean=np.array([.25*8,0.,0.]);drift=np.linalg.solve(A,(F-np.eye(3))@Bmean)
 for t in range(NT):
  active=t*DT>=on-1e-10;new=state@F.T+noise[:,t]@L.T
  if mode=='source_mean':new+=active[:,None]*drift
  if mode=='beta_loss':
   alt=state@Ff.T+noise[:,t]@Lf.T;new[active,-1]=alt[active,-1]
  state=new;out[:,t+1]=state
 return out,on

def maxima(sc):return np.minimum(sc[:,1:],sc[:,:-1]).max(1)
def alarm(sc,threshold):
 hit=(sc[:,1:]>threshold)&(sc[:,:-1]>threshold);has=hit.any(1);return np.where(has,(hit.argmax(1)+2)*.25,np.nan)

def main():
 traininfo=json.loads((OUT/'training.json').read_text());rows=[];models=[];ex={}
 for ni,sigma in enumerate(SIGMAS):
  normal=simulate(dict(kind='indirect',a=.25,c=2.,alpha=1.,beta=1.,sigma=sigma),30,SEEDS['timing_train']+ni,stationary=True);bank=library(traininfo[ni]['qx'],NOBS);I=periodogram(normal['x'][:,NBURN:]).mean(0);ll=-I@bank['inv']-bank['logsum'];ll[bank['classes']!=2]=-np.inf;theta=bank['meta'][ll.argmax()];cal,_=data(WARNING_NCAL,sigma,SEEDS['timing_cal']+ni,'null');sc,_,_,_=estimate(cal[:,:,-1],theta);th={k:float(np.quantile(maxima(v),.95,method='higher')) for k,v in sc.items()};models.append(dict(noise=sigma,theta=theta.tolist(),thresholds=th,known_indirect_family=True))
  for co in COHORTS:
   seed=int(np.random.SeedSequence([833000,ni,co]).generate_state(1)[0]);null,on=data(WARNING_N,sigma,seed,'null');_,en,_,_=estimate(null[:,:,-1],theta)
   for mode in ['null','source_mean','beta_loss']:
    d,tf=data(WARNING_N,sigma,seed,mode)
    if mode=='beta_loss':assert np.allclose(d[:,:,:2],null[:,:,:2],rtol=0,atol=1e-12)
    score,errors,mf,ms=estimate(d[:,:,-1],theta);times={k:alarm(v,th[k]) for k,v in score.items()};ref=times['PredictionError']
    for method,at in times.items():
     for i in range(WARNING_N):
      valid=np.isfinite(at[i]) and at[i]>=tf[i];refvalid=np.isfinite(ref[i]) and ref[i]>=tf[i];late=(np.arange(NT+1)*DT>=tf[i]+8)&(np.arange(NT+1)*DT<tf[i]+12);num=float(np.nanmean(errors[i,late]));den=float(np.nanmean(en[i,late]));online=method!='SourceY_retrospective';rows.append(dict(noise=sigma,cohort=co,trajectory=i,mode=mode,method=method,T_event=tf[i],marker_time=at[i],available_time=at[i] if online else DURATION,online=online,normal_false_alarm=int(np.isfinite(at[i])) if mode=='null' else None,valid=int(valid),early=int(np.isfinite(at[i]) and at[i]<tf[i]),delay=at[i]-tf[i] if valid else np.nan,lead=ref[i]-at[i] if online and valid and refvalid else np.nan,cover_1=int(online and valid and refvalid and ref[i]-at[i]>=1),late_forecast_MSE=num,paired_null_MSE=den,upstream_identical_to_null=bool(mode=='beta_loss')))
    if co==0:
     ex[f'n{ni}_{mode}_states']=d[0];ex[f'n{ni}_{mode}_Yfilter']=mf[0,:,0];ex[f'n{ni}_{mode}_Ysmooth']=ms[0,:,0];ex[f'n{ni}_{mode}_T']=tf[:1]
  print('timing noise complete',sigma,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'timing_metrics.csv.gz',index=False);(OUT/'timing_models.json').write_text(json.dumps(models,indent=2));np.savez_compressed(OUT/'timing_examples.npz',**ex)
if __name__=='__main__':main()
