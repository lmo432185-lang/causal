"""Open latent answers only after ALL_X_ONLY_FROZEN; no model/threshold fitting here."""
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.special import ndtr
from scipy.stats import norm,chi2
from scipy.linalg import eigh
from scipy.signal import lfilter
from settings import *
from model import infer,plant,restore

SQRTPI=np.sqrt(np.pi)
def ks_uniform(u):
 u=np.sort(u.ravel());n=len(u);return float(max(np.max(np.arange(1,n+1)/n-u),np.max(u-np.arange(n)/n)))
def corr(y,m):
 ym=y-y.mean(1,keepdims=True);mm=m-m.mean(1,keepdims=True);return np.sum(ym*mm,1)/np.maximum(np.sqrt(np.sum(ym**2,1)*np.sum(mm**2,1)),1e-20)

def state_scores(y,m,cov,prior,meta,method):
 N,L,_=y.shape;cov=np.broadcast_to(cov,(N,L,2,2));assert np.isfinite(m).all() and np.isfinite(cov).all();v=np.diagonal(cov,axis1=-2,axis2=-1);assert np.min(v)>0;diff=y-m;z=diff/np.sqrt(v);pit=ndtr(z);det=cov[...,0,0]*cov[...,1,1]-cov[...,0,1]*cov[...,1,0];assert np.min(det)>0
 mah=(cov[...,1,1]*diff[...,0]**2+cov[...,0,0]*diff[...,1]**2-(cov[...,0,1]+cov[...,1,0])*diff[...,0]*diff[...,1])/det;nlljoint=np.log(2*np.pi)+.5*np.log(det)+.5*mah;rows=[];case=[]
 for j,s in enumerate(['Y','Z']):
  err=diff[:,:,j]**2;var=v[:,:,j];zz=z[:,:,j];p=pit[:,:,j];std=np.sqrt(var);mse=err.mean(1);crps=std*(zz*(2*p-1)+2*np.exp(-.5*zz**2)/np.sqrt(2*np.pi)-1/SQRTPI);nll=.5*(np.log(2*np.pi*var)+zz**2);r=corr(y[:,:,j],m[:,:,j]);cols=dict(MSE=mse,NMSE=mse/prior[j,j],corr=r,NLL=nll.mean(1),CRPS=crps.mean(1),width90=(2*norm.ppf(.95)*std).mean(1),PIT_mean=p.mean(1))
  for lev in LEVELS:cols[f'cover{int(lev*100)}']=(abs(zz)<=norm.ppf((1+lev)/2)).mean(1)
  for i in range(N):rows.append(dict(**meta,method=method,state=s,trajectory=i,**{k:float(vv[i]) for k,vv in cols.items()}))
  summary={k:float(np.mean(vv)) for k,vv in cols.items()};summary.update(PIT_KS=ks_uniform(p),corr_pooled=float(np.corrcoef(y[:,:,j].ravel(),m[:,:,j].ravel())[0,1]) if np.std(m[:,:,j])>1e-12 else 0.);case.append(dict(**meta,method=method,state=s,**summary))
 cols=dict(MSE=np.mean(np.sum(diff**2,axis=2),axis=1)/2,NMSE=np.mean(np.sum(diff**2/np.diag(prior),axis=2),axis=1)/2,NLL=nlljoint.mean(1),area90=(np.pi*chi2.ppf(.9,2)*np.sqrt(det)).mean(1))
 for lev in LEVELS:cols[f'cover{int(lev*100)}']=(mah<=chi2.ppf(lev,2)).mean(1)
 for i in range(N):rows.append(dict(**meta,method=method,state='joint',trajectory=i,**{k:float(vv[i]) for k,vv in cols.items()}))
 case.append(dict(**meta,method=method,state='joint',**{k:float(np.mean(vv)) for k,vv in cols.items()}))
 bins=[]
 for lo,hi in [(50,99),(100,199),(200,299),(300,399),(400,498)]:
  sl=slice(lo-START,hi-START+1)
  for j,s in enumerate(['Y','Z']):bins.append(dict(**meta,method=method,state=s,start=lo,end=hi,coverage90=float((abs(z[:,sl,j])<=norm.ppf(.95)).mean()),NMSE=float((diff[:,sl,j]**2).mean()/prior[j,j])))
 # Fixed first trajectory and fixed first 1000 whitened errors, no outcome selection.
 e=dict(mean=m[0],sd=np.sqrt(v[0]),error=diff[:5].reshape(-1,2)[:1000],cov=cov[:5].reshape(-1,2,2)[:1000])
 return rows,case,bins,e

def forecast_scores(y,m,var,meta,method):
 var=np.broadcast_to(var,y.shape);z=(y-m)/np.sqrt(var);p=ndtr(z);rows=[];cols=dict(MSE=np.mean((y-m)**2,1),NLL=np.mean(.5*(np.log(2*np.pi*var)+z*z),1),CRPS=np.mean(np.sqrt(var)*(z*(2*p-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/SQRTPI),1),width90=np.mean(2*norm.ppf(.95)*np.sqrt(var),1))
 for lev in LEVELS:cols[f'cover{int(lev*100)}']=(abs(z)<=norm.ppf((1+lev)/2)).mean(1)
 for i in range(len(y)):rows.append(dict(**meta,method=method,state='X_next',trajectory=i,**{k:float(v[i]) for k,v in cols.items()}))
 return rows,dict(**meta,method=method,state='X_next',PIT_KS=ks_uniform(p),**{k:float(np.mean(v)) for k,v in cols.items()})

def theory(meta,params):
 A,Q,R,P=params;d=meta['params']['delay'];n=len(A);w,V=np.linalg.eigh(P);sqrtP=(V*np.sqrt(np.maximum(w,0)))@V.T;Obs=np.array([(np.linalg.matrix_power(A,k)@sqrtP)[2]/np.sqrt(P[2,2]+R) for k in range(30)]);sv=np.linalg.svd(Obs,compute_uv=False);fl=np.zeros_like(P)
 for j in range(2*d):
  Aj=np.linalg.matrix_power(A,j)
  for i,dist in [(0,2*d),(1,d)]:
   if j<dist:fl+=Q[i,i]*np.outer(Aj[:,i],Aj[:,i])
 cov=infer(np.zeros((1,T)),params)['cov'];F=cov[START:499].mean(0);assert np.linalg.eigvalsh(F-fl).min()>-1e-9;rho,evec=eigh(F[:2,:2],P[:2,:2]);gam=[]
 th=meta['params'];theta=[th['a'],th['b'],th['c'],th['alpha']*np.sqrt(th['qy']/th['qz']),th['beta']*np.sqrt(th['qz']),th['qx'],th['obs']**2]
 for lag in DELAYS:
  aa,qq,rr,pp=plant(theta,lag);g=np.array([(np.linalg.matrix_power(aa,k)@pp)[2,2] for k in range(80)]);g[0]+=rr;gam.append(g)
 equiv=float(max(np.max(abs(g-gam[0])) for g in gam));assert equiv<1e-9
 return dict(case=meta['case'],tag=meta['tag'],factor=meta['factor'],value=meta['value'],seed=meta['seed'],augmented_dimension=n,observability_rank=int(np.sum(sv>1e-9)),observability_min_sv=float(sv[-1]),observability_condition=float(sv[0]/max(sv[-1],1e-20)),Y_remaining_variance=float(F[0,0]/P[0,0]),Z_remaining_variance=float(F[1,1]/P[1,1]),Y_unseen_innovation_floor=float(fl[0,0]),Z_unseen_innovation_floor=float(fl[1,1]),Y_filter_variance=float(F[0,0]),Z_filter_variance=float(F[1,1]),best_combo_remaining_fraction=float(rho[0]),worst_combo_remaining_fraction=float(rho[-1]),joint_information_nats=float(.5*np.log(np.linalg.det(P[:2,:2])/np.linalg.det(F[:2,:2]))),delay_X_covariance_max_difference=equiv)

def main():
 mark=json.loads((ROOT/'ALL_X_ONLY_FROZEN.json').read_text());assert mark['cases']==39
 for f,h in mark['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 reg=json.loads((ROOT/'EVALUATION_REGISTERED.json').read_text());assert reg['time']<time.time();assert hashlib.sha256((ROOT/'evaluate.py').read_bytes()).hexdigest()==reg['files']['evaluate.py']
 # First evaluation-time access to private answers; all learners have terminated and outputs are frozen.
 private=json.loads((DATA/'sealed_manifest.json').read_text());rows=[];case_rows=[];bins=[];p0=[];delay_audit=[]
 for meta in private:
  case=meta['case'];dest=OUT/case;x=np.load(DATA/case/'X.npy')[-TEST:];s=np.load(DATA/case/'sealed_states.npz')['states'][-TEST:];gen=meta['generation'];params=(np.array(gen['A']),np.array(gen['Q']),gen['R'],np.array(gen['P0']));A,Q,R,P=params;scale=np.sqrt(np.diag(Q)[:2]);truth=s[:,START:499,:2]/scale;prior=P[:2,:2]/scale[:,None]/scale[None,:];mdata=dict(case=case,tag=meta['tag'],factor=meta['factor'],value=meta['value'],seed=meta['seed']);v=infer(x,params,True);p0.append(theory(meta,params));pred=np.load(dest/'X_only_predictions.npz');methods={}
  for name in ['B_d1','B_d2','B_d3','B_d1_diag','B_current']:
   cv=pred[name+'_cov'];cv=cv[START:499] if cv.ndim==3 else cv;methods[name]=(pred[name+'_mean'][:,START:499],cv)
  methods['B_L1']=(pred['B_L1_mean'][:,START-19:499-19],pred['B_L1_cov'])
  if 'B_boot_mean' in pred:methods['B_boot']=(pred['B_boot_mean'][:,START:499],pred['B_boot_cov'][:,START:499])
  methods['C_filter']=(v['mean'][:,START:499,:2]/scale,v['cov'][START:499,:2,:2]/scale[:,None]/scale[None,:]);methods['C_smoother']=(v['smoothed'][:,START:499,:2]/scale,v['cov_smooth'][START:499,:2,:2]/scale[:,None]/scale[None,:]);methods['C_prior']=(np.zeros_like(truth),prior)
  cg=P[:2,2]/(P[2,2]+R);cv=P[:2,:2]-np.outer(P[:2,2],P[2,:2])/(P[2,2]+R);methods['C_current']=(x[:,START:499,None]*cg/scale,cv/scale[:,None]/scale[None,:]);diag=np.array([np.diag(np.diag(c)) for c in methods['C_filter'][1]]);methods['C_filter_diag']=(methods['C_filter'][0],diag)
  examples=dict(time=np.arange(START,499),truth=truth[0],X=x[0,START:499])
  for name,(m,cov) in methods.items():
   rr,cc,bb,ex=state_scores(truth,m,cov,prior,mdata,name);rows.extend(rr);case_rows.extend(cc);bins.extend(bb)
   for key,val in ex.items():examples[name+'_'+key]=val
  # One-step forecast y[t] uses only observations before t, unlike state filtering.
  ar=json.loads((dest/'AR_models.json').read_text());par=json.loads((dest/'parameters.json').read_text());mu=par['X_training_mean'];ar1=ar['models'][0];pre=lfilter(np.r_[0,ar1['coef']],[1],x-mu,axis=1)+mu
  forecasts={'A_current':(pre[:,START:499],ar1['variance']),'A_AR':(pred['A_AR_forecast'][:,START:499],pred['A_AR_forecast_var']),'B_forecast':(pred['B_forecast'][:,START:499],pred['B_forecast_var'][START:499]),'C_forecast':(v['forecast'][:,START:499],v['forecast_var'][START:499])}
  for name,(m,var) in forecasts.items():
   rr,cc=forecast_scores(x[:,START:499],m,var,mdata,name);rows.extend(rr);case_rows.append(cc)
  for d in [2,3]:
   dm=pred[f'B_d{d}_mean'][:,START:499]-pred['B_d1_mean'][:,START:499];ll0=np.array(par['delay_trajectory_LL']['1']);ll1=np.array(par['delay_trajectory_LL'][str(d)]);delay_audit.append(dict(**mdata,alternative_delay=d,X_loglik_max_difference=float(np.max(abs(ll1-ll0))),Y_posterior_mean_RMS_difference=float(np.sqrt(np.mean(dm[:,:,0]**2))),Z_posterior_mean_RMS_difference=float(np.sqrt(np.mean(dm[:,:,1]**2))),physical_delay_identified=False))
  np.savez_compressed(dest/'evaluation_examples.npz',**examples);print('EVALUATED',case,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'trajectory_metrics.csv.gz',index=False);pd.DataFrame(case_rows).to_csv(OUT/'case_metrics.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'time_calibration.csv',index=False);pd.DataFrame(p0).to_csv(OUT/'observability.csv',index=False);pd.DataFrame(delay_audit).to_csv(OUT/'delay_ambiguity.csv',index=False);(ROOT/'EVALUATION_DONE.json').write_text(json.dumps(dict(time=time.time(),datasets=len(private),trajectory_rows=len(rows),case_rows=len(case_rows),parameters_changed=False),indent=2));print('ALL EVALUATED',len(rows),flush=True)
if __name__=='__main__':main()
