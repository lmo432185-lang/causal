"""Pure-observation Gaussian models; factor count is distinct from memory order."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov,solve_toeplitz
from scipy.optimize import nnls
from settings import *
import native

PARAMS={'obs_ar':2,'factor1':3,'memory2':4,'memory3':5,'factor2':5}

def acov(x,lag=220):
 n,T=x.shape;fft=np.fft.rfft(x,n=2*T,axis=1);s=np.fft.irfft(fft*np.conj(fft),n=2*T,axis=1)[:,:lag+1];return s.sum(0)/(n*T)

def clip_roots(phi):
 roots=np.roots(np.r_[1.,-np.asarray(phi)]);bad=abs(roots)>.9995
 if np.any(bad):roots[bad]*=.9995/abs(roots[bad]);phi=-np.poly(roots)[1:].real
 return phi

def finish(m):
 k=len(m['C']);m['a']=m.get('a',0.);m['P']=solve_discrete_lyapunov(m['F'],m['Q']) if k else np.empty((0,0));m['state_dimension']=k;m['factor_count']=0 if m['kind']=='obs_ar' else 2 if m['kind']=='factor2' else 1;return m

def phi_moments(g,p):
 # Exclude lag0 measurement-noise contamination.
 k=np.arange(p+1,min(len(g),32));A=np.stack([g[k-j] for j in range(1,p+1)],-1);phi=np.linalg.lstsq(A,g[k],rcond=1e-7)[0];return clip_roots(phi)

def two_roots(g):
 phi=phi_moments(g,2);r=np.roots(np.r_[1.,-phi]);valid=bool(np.max(abs(r.imag))<1e-6 and np.min(r.real)>-.9995 and np.max(r.real)<.9995 and abs(r[0]-r[1])>.005)
 return np.sort(np.clip(r.real,-.98,.98))[::-1],valid

def initial(x,kind,seed,start=0,poles=None):
 g=acov(x,50);v=float(g[0]);rng=np.random.default_rng(seed+start*113)
 if kind=='obs_ar':return finish(dict(kind=kind,F=np.empty((0,0)),Q=np.empty((0,0)),C=np.empty(0),a=float(np.clip(g[1]/g[0],-.9995,.9995)),R=max(float(g[0]-g[1]**2/g[0]),1e-8)))
 if kind=='factor1':
  r=float(np.clip(g[2]/g[1] if abs(g[1])>1e-8 else .5,-.98,.98)) if start==0 else float(rng.uniform(-.8,.98));su=np.clip(g[1]/r if abs(r)>.05 else .5*v,.05*v,.95*v);return finish(dict(kind=kind,F=np.array([[r]]),Q=np.array([[max(su*(1-r*r),1e-7)]]),C=np.ones(1),R=max(v-su,.01*v)))
 if kind=='factor2':
  roots=two_roots(g)[0] if poles is None else np.asarray(poles)
  if start and poles is None:roots=np.sort(rng.uniform(-.9,.98,2))[::-1]
  roots=np.clip(roots,-.999,.999);lags=np.arange(1,31);A=roots[None,:]**lags[:,None];amp=nnls(A,g[lags])[0];amp=np.maximum(amp,.01*v);amp*=min(1.,.95*v/amp.sum());Q=np.diag(np.maximum(amp*(1-roots*roots),1e-8));return finish(dict(kind=kind,F=np.diag(roots),Q=Q,C=np.ones(2),R=max(v-amp.sum(),.01*v)))
 p=2 if kind=='memory2' else 3;phi=phi_moments(g,p)
 if start:
  radius=float(rng.uniform(.65,.98));omega=float(rng.uniform(.08,.8))*np.pi;roots=[radius*np.exp(1j*omega),radius*np.exp(-1j*omega)]
  if p==3:roots.append(float(rng.uniform(-.7,.9)))
  phi=-np.poly(roots)[1:].real
 F=np.zeros((p,p));F[0]=phi;F[1:,:-1]=np.eye(p-1);Q=np.zeros((p,p));Q[0,0]=1.;P=solve_discrete_lyapunov(F,Q);latent_v=.7*v;Q[0,0]=latent_v/max(P[0,0],1e-8);return finish(dict(kind=kind,F=F,Q=Q,C=np.r_[1.,np.zeros(p-1)],R=.3*v))

def em(x,m,maxiter=EM_MAX):
 if m['kind']=='obs_ar':m['history']=[];m['converged']=True;return m
 N,T=x.shape;k=len(m['C']);C=m['C'];history=[];small=0;projections=0;kind=m['kind']
 for iteration in range(maxiter):
  p=native.run(x,m,smooth=True);z=p['smean'];S=p['scov'];Lag=p['lag'];ll=float(p['ll'].mean());history.append(ll)
  if len(history)>1 and 0<=history[-1]-history[-2]<TOL:small+=1
  else:small=0
  if small>=5:break
  prev=z[:,:-1].reshape(-1,k);cur=z[:,1:].reshape(-1,k);S0=prev.T@prev+N*S[:-1].sum(0);S1=cur.T@cur+N*S[1:].sum(0);B=cur.T@prev+N*Lag.sum(0)
  if kind in ['factor1','factor2']:
   roots=np.clip(np.diag(B)/np.maximum(np.diag(S0),1e-12),-.9995,.9995);F=np.diag(roots);q=(np.diag(S1)-2*roots*np.diag(B)+roots*roots*np.diag(S0))/(N*(T-1));Q=np.diag(np.maximum(q,1e-8))
   if kind=='factor2':
    ix=np.argsort(roots)[::-1];F=F[ix][:,ix];Q=Q[ix][:,ix]
  else:
   phi=np.linalg.pinv(S0,rcond=1e-8)@B[0];oldphi=phi.copy();phi=clip_roots(phi);projections+=int(np.max(abs(phi-oldphi))>1e-10);F=np.zeros((k,k));F[0]=phi;F[1:,:-1]=np.eye(k-1);q=(S1[0,0]-2*phi@B[0]+phi@S0@phi)/(N*(T-1));Q=np.zeros((k,k));Q[0,0]=max(float(q),1e-8)
  obsmean=z@C;R=max(float(np.mean((x-obsmean)**2)+np.mean(np.einsum('i,tij,j->t',C,S,C))),1e-6)
  m=finish(dict(kind=kind,F=F,Q=Q,C=C,a=0.,R=R))
 m['history']=history;m['converged']=small>=5;m['iterations']=len(history);m['stability_projections']=projections;return m

def fit(x,kind,seed,poles=None):
 scale=max(float(np.sqrt(np.mean(x*x))),1e-8);xx=x/scale;allmodels=[]
 for start in range(1 if kind=='obs_ar' else 3):
  pp=None if poles is None else poles[start]
  if pp is None and poles is not None:continue
  m=em(xx,initial(xx,kind,seed,start,poles=pp));ll=float(native.run(xx,m)['ll'].mean());allmodels.append((ll,start,m))
 if not allmodels:return None
 ll,start,m=max(allmodels,key=lambda z:z[0]);m['scale']=scale;m['best_start']=start;m['train_mean_loglik']=ll-np.log(scale);m['start_logliks']=[float(z[0]) for z in allmodels];return m

def parent_proposals(x,parent,seed):
 xx=x/parent['scale'];p=native.run(xx,parent,smooth=True);rows=[];meta=[]
 for i in range(3):
  draw=native.sample(parent,p,seed+i*919,draws=2);w=(draw@parent['C']).reshape(-1,x.shape[1]);r,valid=two_roots(acov(w,50));rows.append(r if valid else None);meta.append(dict(valid=valid,proposed_poles=r.tolist(),posterior_mean_variance=float(np.mean(np.einsum('i,tij,j->t',parent['C'],p['scov'],parent['C']))),draws=2))
 return rows,meta

def pack(x):
 if isinstance(x,np.ndarray):return {'array':x.tolist(),'shape':list(x.shape)}
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,list):return [pack(v) for v in x]
 if isinstance(x,(np.integer,np.floating)):return x.item()
 return x

def unpack(x):
 if isinstance(x,dict) and set(x)=={'array','shape'}:return np.array(x['array'],float).reshape(x['shape'])
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 if isinstance(x,list):return [unpack(v) for v in x]
 return x
