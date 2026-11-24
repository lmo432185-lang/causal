"""X-only proposal and joint Gaussian inference. No generator or truth imports."""
import copy,time,numpy as np
from scipy.linalg import solve_discrete_lyapunov,block_diag,null_space
from scipy.signal import lfilter
from settings import *
from model import acov,two_roots,phi_moments
import native

def finish(m):
 m['P']=solve_discrete_lyapunov(m['F'],m['Q']);m['state_dimension']=len(m['C']);m['kind']='extension';m['a']=float(m.get('a',0));assert np.isfinite(m['P']).all();return m

def psd(Q,floor=1e-7):
 v,U=np.linalg.eigh((Q+Q.T)/2);return (U*np.maximum(v,floor))@U.T

def residual(x,base):
 p=native.run(x/base['scale'],base);mu=np.zeros_like(x);mu[:,1:]=np.einsum('i,nti->nt',base['C']@base['F'],p['mean'][:,:-1])+base.get('a',0)*x[:,:-1]/base['scale'];return x/base['scale']-mu,p

def proposals(x,base,seed):
 e,p=residual(x,base);ge=acov(e[:,200:],50);rres=float(np.clip(ge[1]/max(ge[0],1e-12),-.98,.98));smooth=native.run(x/base['scale'],base,smooth=True);paths=native.sample(base,smooth,seed,draws=2);w=(paths@base['C']).reshape(-1,x.shape[1]);roots,valid=two_roots(acov(w,50));rr=float(roots[-1]);return dict(residual_r=rres,residual_variance=float(ge[0]),path_r=rr,path_valid=valid,path=paths[0],innovation=e)

def init_model(base,mode,route,proposal,x,seed,start):
 rng=np.random.default_rng(seed+start*1009);m=copy.deepcopy(base);d=len(base['C']);add=mode in ['latent','both','dense'];m['mode']=mode;m['route']=route;m['base_dimension']=d;m['a']=0.;m['scale']=base['scale'];m['converged']=False
 if add:
  if route=='path' and not proposal['path_valid']:return None
  if route=='direct':
   g=acov(x/base['scale'],50);r=float(np.clip(g[2]/g[1] if abs(g[1])>.01 else 0.,-.95,.95)) if start==0 else [-.7,.65][start-1]
  elif route=='path':r=float(np.clip(proposal['path_r']+(start-1)*.08,-.98,.98))
  else:r=float(np.clip(proposal['residual_r']+(start-1)*.12,-.98,.98))
  q=max(.05*proposal['residual_variance']*(1-r*r),1e-5);m['F']=block_diag(base['F'],np.array([[r]]));m['Q']=block_diag(base['Q'],np.array([[q]]));m['C']=np.r_[base['C'],1.]
 if mode=='dense':
  if route=='direct':
   k=d+1;g=acov(x/base['scale'],50);phi=phi_moments(g,k);F=np.zeros((k,k));F[0]=phi;F[1:,:-1]=np.eye(k-1)
   if start:F=F*(.85 if start==1 else .65)
   Q=np.zeros((k,k));Q[0,0]=.1;P=solve_discrete_lyapunov(F,Q);Q/=max(P[0,0],1e-8);Q*=.7;unit=m['C']/np.linalg.norm(m['C']);V=np.c_[unit,null_space(unit[None])];T=V/np.linalg.norm(m['C']);m['F']=T@F@np.linalg.inv(T);m['Q']=psd(T@Q@T.T);m['R']=.3
  else:
   z=proposal['path'];e=proposal['innovation'];new=lfilter([.25],[1.,-r],e,axis=1);joined=np.concatenate([z,new[:,:,None]],2);prev=joined[:,:-1].reshape(-1,d+1);cur=joined[:,1:].reshape(-1,d+1);F=np.linalg.lstsq(prev,cur,rcond=1e-6)[0].T;radius=max(abs(np.linalg.eigvals(F)));F*=min(1.,.98/max(radius,1e-10));u=cur-prev@F.T;m['F']=F;m['Q']=psd(np.cov(u.T)+.01*np.eye(d+1));m['R']=max(base['R']*.9,.01)
 return finish(m)

def train(x,m,maxiter=NEW_ITERS):
 if m is None:return None
 m=copy.deepcopy(m);xx=x/m['scale'];N,T=xx.shape;k=len(m['C']);d=m['base_dimension'];history=[];small=0;projections=0;mode=m['mode']
 for it in range(maxiter):
  p=native.run(xx,m,smooth=True);z=p['smean'];S=p['scov'];history.append(float(p['ll'][:,200:].mean()))
  if len(history)>1 and 0<=history[-1]-history[-2]<1e-6:small+=1
  else:small=0
  if small>=5:break
  if mode=='dense':
   u=z[:,:-1].reshape(-1,k);v=z[:,1:].reshape(-1,k);A=u.T@u+N*S[:-1].sum(0);B=v.T@u+N*p['lag'].sum(0);D=v.T@v+N*S[1:].sum(0);F=B@np.linalg.pinv(A,rcond=1e-8);rad=max(abs(np.linalg.eigvals(F)));projections+=int(rad>.9995);F*=min(1.,.9995/max(rad,1e-10));Q=(D-F@B.T-B@F.T+F@A@F.T)/(N*(T-1));m['F']=F;m['Q']=psd(Q)
  elif mode in ['latent','both']:
   u=z[:,:-1,-1];v=z[:,1:,-1];A=np.sum(u*u)+N*S[:-1,-1,-1].sum();B=np.sum(u*v)+N*p['lag'][:,-1,-1].sum();D=np.sum(v*v)+N*S[1:,-1,-1].sum();r=float(np.clip(B/max(A,1e-12),-.9995,.9995));m['F'][-1,-1]=r;m['Q'][-1,-1]=max(float((D-2*r*B+r*r*A)/(N*(T-1))),1e-7)
  emission=z@m['C']
  if mode in ['obs','both']:m['a']=float(np.clip(np.sum(xx[:,:-1]*(xx[:,1:]-emission[:,1:]))/np.sum(xx[:,:-1]**2),-.95,.95))
  e=xx-emission;e[:,1:]-=m['a']*xx[:,:-1];m['R']=max(float(np.mean(e[:,1:]**2)+np.mean(np.einsum('i,tij,j->t',m['C'],S[1:],m['C']))),1e-6);m=finish(m)
 m['iterations']=len(history);m['history']=history;m['converged']=small>=5;m['stability_projections']=projections;m['train_loglik']=float(native.run(xx,m)['ll'][:,200:].mean());return m

def fit_candidate(x,base,mode,route,proposal,seed):
 tic=time.process_time();fits=[];fail=[]
 for start in range(NEW_STARTS):
  try:
   m=train(x,init_model(base,mode,route,proposal,x,seed,start))
   if m is not None:fits.append((m['train_loglik'],m))
   else:fail.append('invalid_path')
  except (ValueError,AssertionError,np.linalg.LinAlgError,FloatingPointError) as ex:fail.append(type(ex).__name__)
 if not fits:return None,dict(cpu=time.process_time()-tic,failures=fail)
 m=max(fits,key=lambda z:z[0])[1];return m,dict(cpu=time.process_time()-tic,failures=fail,start_logliks=[z[0] for z in fits])
