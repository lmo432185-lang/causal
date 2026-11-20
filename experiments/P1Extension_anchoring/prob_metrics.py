import numpy as np
from scipy.special import ndtr
from scipy.stats import norm,chi2
START=50
LEVELS=[.5,.8,.9,.95]
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

