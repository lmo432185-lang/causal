"""X-only EM/Kalman probability-state models, direct and posterior-path proposals."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import logsumexp
from settings import *
import native

def spd(c,floor=1e-6):
 c=(c+c.T)/2;w,v=np.linalg.eigh(c);return (v*np.maximum(w,max(float(np.max(w)),1.)*floor))@v.T

def stable(F):
 r=max(abs(np.linalg.eigvals(F)),default=0);return F*(.995/r) if r>.995 else F

def complete(m):
 k=len(m['C']);m['P']=spd(solve_discrete_lyapunov(m['F'],m['Q'])) if k else np.empty((0,0));return m

def moments(z,x):
 # z contains full posterior path draws when initialized recursively.
 n,T,k=z.shape;z=z/max(1e-6,float(np.std(z))) if k==1 else z/np.maximum(z.std((0,1)),1e-6);prev=z[:,:-1].reshape(-1,k);cur=z[:,1:].reshape(-1,k);F=stable(np.linalg.lstsq(prev,cur,rcond=1e-7)[0].T);q=cur-prev@F.T;Q=spd(q.T@q/len(q),1e-4);D=np.concatenate([x[:,:-1,None],z[:,1:]],-1).reshape(-1,k+1);y=x[:,1:].ravel();b=np.linalg.lstsq(D,y,rcond=1e-7)[0];res=y-D@b;return complete(dict(F=F,Q=Q,C=b[1:],a=float(np.clip(b[0],-.99,.99)),R=max(float(np.mean(res*res)),1e-4*float(np.var(x)))))

def cva(past,future,current,k):
 def white(c):
  w,v=np.linalg.eigh((c+c.T)/2);return (v/np.sqrt(np.maximum(w,max(w[-1]*1e-6,1e-8))))@v.T
 past=past.reshape(-1,past.shape[-1]);future=future.reshape(-1,future.shape[-1]);b=np.c_[np.ones(current.size),current.ravel()];g=np.linalg.lstsq(b,past,rcond=1e-7)[0];gy=np.linalg.lstsq(b,future,rcond=1e-7)[0];a=past-b@g;y=future-b@gy;W=white(a.T@a/len(a));Y=white(y.T@y/len(y));u,sv,_=np.linalg.svd(W@(a.T@y/len(y))@Y,full_matrices=False);E=W@u[:,:k];return (a@E),dict(g=g,E=E,canonical=sv)

def initial(x,k):
 if k==0:
  a=float(np.sum(x[:,1:]*x[:,:-1])/max(np.sum(x[:,:-1]**2),1e-10));r=x[:,1:]-a*x[:,:-1];return dict(F=np.empty((0,0)),Q=np.empty((0,0)),C=np.empty(0),P=np.empty((0,0)),a=float(np.clip(a,-.99,.99)),R=max(float(np.mean(r*r)),1e-8))
 t=np.arange(40,140);past=np.stack([x[:,t-j] for j in range(1,41)],-1);future=np.stack([x[:,t+j] for j in range(1,21)],-1);z,_=cva(past,future,x[:,t],k);return moments(z.reshape(len(x),len(t),k),x[:,t])

def em(x,m,iters=EM_ITERS):
 if not len(m['C']):return m
 x=np.asarray(x,float);N,T=x.shape;k=len(m['C']);history=[]
 for iteration in range(iters):
  p=native.run(x,m,smooth=True);z=p['smean'];S=p['scov'];lag=p['lag'];history.append(float(p['ll'].mean()))
  prev=z[:,:-1].reshape(-1,k);cur=z[:,1:].reshape(-1,k);Sprev=prev.T@prev+N*S[:-1].sum(0);Scur=cur.T@cur+N*S[1:].sum(0);Scross=cur.T@prev+N*lag.sum(0);F=stable(Scross@np.linalg.pinv(Sprev));Q=spd((Scur-F@Scross.T-Scross@F.T+F@Sprev@F.T)/(N*(T-1)),1e-5)
  obs=x[:,1:].ravel();xp=x[:,:-1].ravel();crossz=z[:,1:].reshape(-1,k);G=np.empty((k+1,k+1));G[0,0]=xp@xp;G[0,1:]=xp@crossz;G[1:,0]=G[0,1:];G[1:,1:]=Scur;rhs=np.r_[xp@obs,crossz.T@obs];b=np.linalg.pinv(G)@rhs;b[0]=np.clip(b[0],-.995,.995);R=max(float((obs@obs-2*b@rhs+b@G@b)/len(obs)),float(np.var(x))*1e-5,1e-8);m=complete(dict(F=F,Q=Q,C=b[1:],a=float(b[0]),R=R))
 m['EM_loglik']=history;return m

def recursive(x,old,k,seed):
 p=native.run(x,old,smooth=True);draw=native.sample(old,p,seed,draws=2);S=np.diagonal(p['scov'],axis1=1,axis2=2).mean(0);priority=S/np.maximum(np.var(p['smean'],axis=(0,1)),1e-6)+np.var(np.diff(p['smean'],axis=1),axis=(0,1));parent=int(np.argmax(priority));w=draw[:,:,:,parent].reshape(-1,x.shape[1]);t=np.arange(40,140);past=np.stack([w[:,t-j] for j in range(1,21)],-1);future=np.stack([w[:,t+j] for j in range(1,21)],-1);a,enc=cva(past,future,w[:,t],1);a=a.reshape(2,len(x),len(t),1);oldpaths=draw[:,:,t,:];joint=np.concatenate([oldpaths,a],-1).reshape(2*len(x),len(t),k);xx=np.tile(x[:,t],(2,1));m=moments(joint,xx);return m,dict(parent=parent,priority=priority.tolist(),posterior_variance=S.tolist(),sample_draws=2,candidate_variance=float(np.var(a)),new_joint_dimension=k)

def future_distribution(x,m,post,rolling=False):
 k=len(m['C']);N=len(x);O=len(ORIGINS);mean=np.empty((N,O,H));cov=np.empty((O,H,H));a=m['a'];C=m['C'];F=m['F'];Q=m['Q'];A=np.zeros((k+1,k+1));A[:k,:k]=F;A[k,:k]=C@F;A[k,k]=a;V=np.zeros_like(A);V[:k,:k]=Q;V[:k,k]=Q@C;V[k,:k]=C@Q;V[k,k]=C@Q@C+m['R'];powers=[np.eye(k+1)]
 for h in range(H):powers.append(powers[-1]@A)
 for oi,t in enumerate(ORIGINS):
  if rolling:
   pp=native.run(x[:,t-40:t+1],m);mu=pp['mean'][:,-1];P=pp['cov'][-1]
  else:mu=post['mean'][:,t];P=post['cov'][t]
  z=np.c_[mu,x[:,t]];Paug=np.zeros((k+1,k+1));Paug[:k,:k]=P;Ps=[]
  for h in range(H):
   z=z@A.T;Paug=A@Paug@A.T+V;Ps.append(Paug.copy());mean[:,oi,h]=z[:,-1]
  for h in range(H):
   for j in range(h,H):cov[oi,h,j]=cov[oi,j,h]=(Ps[h]@powers[j-h].T)[-1,-1]
 return mean,cov

def component(x,m,update=True,rolling=False):
 scale=m.get('scale',1.);xx=np.asarray(x,float)/scale;p=native.run(xx,m,stop=100000 if update else 40);p['ll']-=np.log(scale);mu,cov=future_distribution(xx,m,p,rolling);return dict(mean=mu*scale,cov=cov*scale**2,post=p)

def target(x):return np.stack([x[:,np.array(ORIGINS)+h] for h in range(1,H+1)],-1)
def logpdf(mu,cov,y):
 iv=np.linalg.inv(cov);r=y-mu;return -.5*(np.einsum('noi,oij,noj->no',r,iv,r)+np.linalg.slogdet(cov)[1]+y.shape[-1]*np.log(2*np.pi))
def selection_score(x,m):
 c=component(x,m);return -logpdf(c['mean'],c['cov'],target(x)).mean(1)/H

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
