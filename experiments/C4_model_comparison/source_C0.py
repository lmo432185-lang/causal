import os
os.environ['MPLCONFIGDIR']='/workspace/work/mpl';os.environ['XDG_CACHE_HOME']='/workspace/work/cache'
from pathlib import Path
import numpy as np,json,csv,gzip,time
from scipy.fft import rfft,irfft,next_fast_len
from scipy.special import expit
from source_A import simulate,fit_normal,statistics,DT,SIGMAS,TAUS,STRENGTHS
from support import fb,first_run
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
TAUGRID=np.geomspace(.05,5,41);FORECAST=[1,10,40,100];CUT=300
CONFIG=dict(experiment='C0-A structured source/transfer inference; limited C0-B identifiability audit',dt=DT,duration=25,test_each_cell=200,noise=SIGMAS,tau_test=TAUS,strength_test=STRENGTHS,source_test_onset=10,normal_train=2000,unlabelled_mixture_train=200,calibration_null=500,test_null=500,tau_grid=TAUGRID.tolist(),onset_grid='0.5 to observed endpoint minus 0.25, step 0.05; no T1 supplied',models={'M1':'frozen seed11 HMM from A/B','M2':'two explicit states with one shared tau learned from independent X only','M3':'two explicit states with per-trajectory unknown tau, onset and positive amplitude profile likelihood'},known_assumptions=['one persistent positive hidden source step','first-order transfer','linear observation drift, additive Gaussian process noise','no unmodelled pure transport dead time','effect gain normalized to 1; physical latent scale not identified'],unknown_learned=['normal AR coefficients and innovation variance from independent normal X','source onset','source effect amplitude','shared or individual tau'],true_Y_U_tau_T1_not_in_fit=True,intervals='approximate 95% 1D profile likelihood intervals, 2deltaLL<=3.841; not Bayesian intervals; calibrated coverage evaluated',forecast_cut=15,forecast_horizons=[h*DT for h in FORECAST],false_alarm_target=.05)

def mixture(n,sigma,seed):
 rng=np.random.default_rng(seed);on=rng.uniform(3,18,n);on=np.round(on/DT).astype(int);tau=np.exp(rng.uniform(np.log(.05),np.log(5),n));amp=rng.uniform(.5,1.5,n);amp[rng.random(n)<.3]=0;x=np.zeros((n,501));u=np.zeros(n);x[:,0]=rng.normal(10,.3,n);q=np.exp(-.15*DT);r=np.exp(-DT/tau);j=(r-q)/(.15-1/tau);k=(1-q)/.15
 for t in range(500):
  y=amp*(t>=on);x[:,t+1]=q*x[:,t]+1.5*k-1.5*(y*k+(u-y)*j)+sigma*np.sqrt((1-q*q)/.3)*rng.normal(size=n);u=y+(u-y)*r
 return x

def templates(T,base,taus=TAUGRID):
 a=-np.log(base['q'])/DT;q=base['q'];k=(1-q)/a;ages=np.arange(T)*DT;weights=[]
 for tau in taus:
  j=(np.exp(-DT/tau)-q)/(a-1/tau)
  if abs(a-1/tau)<1e-8:j=DT*q
  weights.append(k-np.exp(-ages/tau)*j)
 return np.array(weights)

def bank(x,base):
 T=x.shape[1]-1;starts=np.arange(10,T-4);w=templates(T,base);length=next_fast_len(2*T-1);res=-(x[:,1:]-base['q']*x[:,:-1]-base['c']);rf=rfft(res,n=length,axis=1);score=np.empty((len(x),len(TAUGRID),len(starts)));amps=np.empty_like(score)
 for k,h in enumerate(w):
  dot=irfft(rf*rfft(h[::-1],n=length)[None,:],n=length,axis=1)[:,T-1+starts];norm=np.cumsum(h*h)[T-starts-1];amps[:,k]=np.maximum(dot,0)/norm;score[:,k]=np.maximum(dot,0)**2/(norm*base['var'])
 return dict(score=score,amps=amps,starts=starts,T=T)

def infer(b,threshold,fixed=None):
 score=b['score'] if fixed is None else b['score'][:,fixed:fixed+1];n=len(score);flat=score.reshape(n,-1);best=np.argmax(flat,1);ki,si=np.unravel_index(best,score.shape[1:]);ti=ki if fixed is None else np.full(n,fixed);ii=np.arange(n);maximum=flat[ii,best];detected=maximum>threshold;amp=b['amps'][ii,ti,si];on=b['starts'][si]*DT;tau=TAUGRID[ti];onprof=score.max(1);taumax=score.max(2);keep_on=onprof>=maximum[:,None]-3.841459;keep_tau=taumax>=maximum[:,None]-3.841459
 olow=np.min(np.where(keep_on,b['starts'][None,:]*DT,np.inf),1);ohi=np.max(np.where(keep_on,b['starts'][None,:]*DT,-np.inf),1)
 if fixed is None:
  tlow=np.min(np.where(keep_tau,TAUGRID[None,:],np.inf),1);thi=np.max(np.where(keep_tau,TAUGRID[None,:],-np.inf),1)
 else:tlow=np.full(n,np.nan);thi=tlow.copy()
 return dict(on=on,tau=tau,amp=amp,detected=detected,score=maximum,onlo=olow,onhi=ohi,taulo=tlow,tauhi=thi,tau_boundary=(ti==0)|(ti==len(TAUGRID)-1))

def reconstruct(est,T=501):
 age=np.maximum(np.arange(T)[None,:]*DT-est['on'][:,None],0);amp=est['amp']*est['detected'];u=amp[:,None]*(1-np.exp(-age/est['tau'][:,None]));y=amp[:,None]*(np.arange(T)[None,:]*DT>=est['on'][:,None]);return y,u

def forecast(est,last,base):
 age=np.maximum(15-est['on'],0);amp=est['amp']*est['detected'];a=-np.log(base['q'])/DT;pred=[]
 for h in FORECAST:
  delta=h*DT;q=base['q']**h;j=(np.exp(-delta/est['tau'])-q)/(a-1/est['tau']);k=(1-q)/a
  pred.append(q*last+base['c']*(1-q)/(1-base['q'])-amp*(k-np.exp(-age/est['tau'])*j))
 return np.stack(pred,1)

def old_inference(x,p,thr,base):
 e,prior,post,back,g,ll=fb(x,p);prob=g[:,:,0];on=np.full(len(x),np.nan)
 for i,v in enumerate(prob):
  first=first_run(v>expit(thr))
  if first is not None:on[i]=first*DT
 a=-np.log(base['q'])/DT;k=(1-base['q'])/a;mu=prob*p['mu'][0]+(1-prob)*p['mu'][1];force=np.maximum(((base['q']-1-p['beta'])*x[:,:-1]+base['c']-mu)/k,0);detected=np.isfinite(on);force*=detected[:,None]
 return dict(on=on,detected=detected,u=force,prob=prob)

def old_forecast(x,p):
 post=fb(x,p)[2][:,-1];xx=x[:,-1].copy();pred=[]
 for h in range(1,max(FORECAST)+1):
  post=post@p['A'];xx=(1+p['beta'])*xx+post@p['mu']
  if h in FORECAST:pred.append(xx.copy())
 return np.stack(pred,1)

def evaluate(data,est,model,meta,pred,true_tau,physical_gain=1.5):
 rows=[];trueu=data['u'];truey=data['y'];n=len(trueu)
 if model=='M1':yhat=None;uhat=est['u']/physical_gain
 else:yforce,uforce=reconstruct(est);yhat=yforce/physical_gain;uhat=uforce/physical_gain
 for i in range(n):
  det=bool(est['detected'][i]);on=float(est['on'][i]);onerr=abs(on-10) if det else np.nan;row=dict(**meta,model=model,trajectory=i,detected=int(det),onset=on if det else np.nan,onset_error=onerr,onset_signed=on-10 if det else np.nan,onset_error_capped=onerr if det else 15.,U_rmse=float(np.sqrt(np.mean((uhat[i,:500]-trueu[i,:500])**2))),Y_rmse=float(np.sqrt(np.mean((yhat[i,:500]-truey[i])**2))) if yhat is not None else np.nan,tau_estimate=float(est['tau'][i]) if model!='M1' and det else np.nan,tau_error=abs(float(est['tau'][i])-true_tau) if model!='M1' and det else np.nan,onset_interval_coverage=int(est['onlo'][i]<=10<=est['onhi'][i]) if model!='M1' and det else np.nan,onset_interval_width=float(est['onhi'][i]-est['onlo'][i]) if model!='M1' and det else np.nan,tau_interval_coverage=int(est['taulo'][i]<=true_tau<=est['tauhi'][i]) if model=='M3' and det else np.nan,tau_interval_width=float(est['tauhi'][i]-est['taulo'][i]) if model=='M3' and det else np.nan,tau_boundary=int(est['tau_boundary'][i]) if model=='M3' and det else np.nan)
  for j,h in enumerate(FORECAST):row[f'forecast_mse_h{h}']=float((pred[i,j]-data['x'][i,CUT+h])**2)
  rows.append(row)
 return rows

def writecsv(path,rows):
 with gzip.open(path,'wt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2));oldcal=json.loads((OUT/'calibration_A.json').read_text());cal={};rows=[];nullrows=[];examples={};start=time.time()
 for gi,sigma in enumerate(SIGMAS):
  base=fit_normal(simulate(2000,sigma,1,0,21000+gi,False)['x']);train=mixture(200,sigma,22000+gi);tr=bank(train,base);fixed=int(np.argmax(tr['score'].max(2).sum(0)));tau_shared=float(TAUGRID[fixed]);del tr
  val=simulate(500,sigma,1,0,23000+gi,False)['x'];testnull=simulate(500,sigma,1,0,24000+gi,False)['x'];thresholds={};vb=bank(val,base);vbp=bank(val[:,:CUT+1],base)
  for model,idx in [('M2',fixed),('M3',None)]:
   threshold_value=np.quantile(vb['score'][:,idx].max(1) if idx is not None else vb['score'].max((1,2)),.95,method='higher');threshold_prefix=np.quantile(vbp['score'][:,idx].max(1) if idx is not None else vbp['score'].max((1,2)),.95,method='higher');thresholds[model]=dict(full=float(threshold_value),prefix=float(threshold_prefix))
  del vb,vbp
  ck=np.load(OUT/f'model_{gi}_11.npz');p={k:ck[k] for k in ['beta','mu','var','A','pi']};thr1=oldcal[str(sigma)]['HMM_thresholds']['11'];nb=bank(testnull,base)
  null_est={'M1':old_inference(testnull,p,thr1,base)}
  for model,idx in [('M2',fixed),('M3',None)]:null_est[model]=infer(nb,thresholds[model]['full'],idx)
  for model,est in null_est.items():
   for i,v in enumerate(est['detected']):nullrows.append(dict(sigma=sigma,model=model,trajectory=i,false_alarm=int(v)))
  cal[str(sigma)]=dict(normal_AR=base,shared_tau=tau_shared,shared_index=fixed,thresholds=thresholds,HMM_threshold=thr1)
  for ti,tau in enumerate(TAUS):
   for si,strength in enumerate(STRENGTHS):
    data=simulate(200,sigma,tau,strength,13000+gi);full=bank(data['x'],base);prefix=bank(data['x'][:,:CUT+1],base);ests={'M1':old_inference(data['x'],p,thr1,base)};preds={'M1':old_forecast(data['x'][:,:CUT+1],p)}
    for model,idx in [('M2',fixed),('M3',None)]:
     ests[model]=infer(full,thresholds[model]['full'],idx);pe=infer(prefix,thresholds[model]['prefix'],idx);preds[model]=forecast(pe,data['x'][:,CUT],base)
    for model,est in ests.items():rows.extend(evaluate(data,est,model,dict(sigma=sigma,tau=tau,strength=strength),preds[model],tau))
    key=f'n{gi}_t{ti}_s{si}_';examples[key+'x']=data['x'][0];examples[key+'Y']=data['y'][0];examples[key+'U']=data['u'][0]
    for model,est in ests.items():
     examples[key+model+'_on']=est['on'][0] if est['detected'][0] else np.nan
     if model!='M1':
      y,u=reconstruct(est);examples[key+model+'_Y']=y[0]/1.5;examples[key+model+'_U']=u[0]/1.5;examples[key+model+'_tau']=est['tau'][0]
     else:examples[key+model+'_U']=est['u'][0]/1.5
  print('noise',sigma,'done',round(time.time()-start,1),'shared_tau',tau_shared,flush=True)
 (OUT/'calibration.json').write_text(json.dumps(cal,indent=2));writecsv(OUT/'per_trajectory.csv.gz',rows);writecsv(OUT/'null_trajectories.csv.gz',nullrows);np.savez_compressed(OUT/'examples.npz',**examples);summarize(rows,nullrows)

def summarize(rows,nullrows):
 out=[]
 for sigma in SIGMAS:
  for tau in TAUS:
   for strength in STRENGTHS:
    for model in ['M1','M2','M3']:
     sub=[r for r in rows if r['sigma']==sigma and r['tau']==tau and r['strength']==strength and r['model']==model];metrics={k:statistics([r[k] for r in sub]) for k in sub[0] if k not in ['sigma','tau','strength','model','trajectory']};out.append(dict(sigma=sigma,tau=tau,strength=strength,model=model,metrics=metrics))
 null=[]
 for sigma in SIGMAS:
  for model in ['M1','M2','M3']:
   vals=[r['false_alarm'] for r in nullrows if r['sigma']==sigma and r['model']==model];null.append(dict(sigma=sigma,model=model,false_alarm=statistics(vals)))
 (OUT/'summary.json').write_text(json.dumps(out,indent=2));(OUT/'null_summary.json').write_text(json.dumps(null,indent=2))
 for r in out:
  if r['sigma']==.35 and r['strength']==1:print(r['tau'],r['model'],{k:r['metrics'][k]['mean'] for k in ['detected','onset_error','U_rmse','tau_error','tau_interval_coverage','forecast_mse_h40']},flush=True)
if __name__=='__main__':main()
