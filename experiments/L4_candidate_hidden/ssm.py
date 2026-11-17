"""Constrained linear-Gaussian EM; inputs contain observations only."""
import numpy as np
from scipy.signal import ss2tf,lfilter
from scipy.optimize import lsq_linear
from scipy.linalg import solve_discrete_lyapunov

def setup(kind,delay=1):
 if kind=='single':return 2,[[0],[0,1]],None
 n=3 if delay==1 or kind=='G1' else 4
 if kind=='G1':mask=[[0],[1],[0,1,2]];buf=None
 elif kind in ['G2','G3']:
  mask=[[0],[0 if delay==1 else 3,1],[1,2] if kind=='G2' else [0,1,2]];buf=0 if n==4 else None
 else:mask=[[0,1 if delay==1 else 3],[1],[0,2]];buf=1 if n==4 else None
 return n,mask,buf

def initialize(kind,delay,seed,oracle=False):
 n,mask,buf=setup(kind,delay);rng=np.random.default_rng(seed);A=np.zeros((n,n));phys=2 if kind=='single' else 3
 for i,cols in enumerate(mask):
  for j in cols:A[i,j]=rng.uniform(.2,.9) if i==j else rng.uniform(.05,.6)
 if buf is not None:A[3,buf]=1.
 q=np.ones(n);q[phys-1]=rng.uniform(.03,.3)
 if n>phys:q[-1]=0.
 r=float(rng.uniform(.03,.3));P=np.eye(n)*5;P[phys-1,phys-1]=1.
 P=solve_discrete_lyapunov(A,np.diag(q))
 return dict(A=A,Q=np.diag(q),r=r,P0=P,kind=kind,delay=delay,oracle=oracle)

def obs_matrix(p):
 n=len(p['A']);ix=1 if p['kind']=='single' else 2
 if p['oracle']:
  H=np.eye(n)[:3];R=np.diag([0.,0.,p['r']])
 else:H=np.eye(n)[ix:ix+1];R=np.array([[p['r']]])
 return H,R

def covariance(p,T):
 A,Q,P=p['A'],p['Q'],p['P0'].copy();H,R=obs_matrix(p);n=len(A);K=[];PP=[];PF=[];SI=[];LD=[];cut=T
 for t in range(T):
  S=H@P@H.T+R;inv=np.linalg.inv(S);k=P@H.T@inv;f=P-k@H@P;f=(f+f.T)/2;K.append(k);PP.append(P.copy());PF.append(f);SI.append(inv);LD.append(np.linalg.slogdet(S)[1]);new=A@f@A.T+Q
  if t>=8 and np.max(abs(new-P))<1e-12 and np.max(abs(k-K[-2]))<1e-12:
   cut=t+1;break
  P=new
 for t in range(cut,T):K.append(K[-1]);PP.append(new.copy());PF.append(PF[-1]);SI.append(SI[-1]);LD.append(LD[-1])
 PP=np.array(PP);PF=np.array(PF);K=np.array(K);SI=np.array(SI);LD=np.array(LD);J=np.empty((T-1,n,n))
 for t in range(min(cut,T-1)):J[t]=PF[t]@A.T@np.linalg.pinv(PP[t+1],rcond=1e-12,hermitian=True)
 if cut<T-1:J[cut:]=J[cut-1]
 PS=PF.copy()
 for t in range(T-2,-1,-1):PS[t]=PF[t]+J[t]@(PS[t+1]-PP[t+1])@J[t].T;PS[t]=(PS[t]+PS[t].T)/2
 return H,K,PP,PF,PS,SI,LD,J,cut

def recur_polynomial(A,B,U,z0=None):
 N,L,m=U.shape;n=len(A);out=np.zeros((N,L,n))
 for j in range(m):
  nums,den=ss2tf(A,B,A,B,input=j)
  for i in range(n):out[:,:,i]+=lfilter(np.real_if_close(nums[i]),np.real_if_close(den),U[:,:,j],axis=1)
 if z0 is not None:
  P=[];cur=A.copy()
  for _ in range(L):P.append(cur.copy());cur=A@cur
  P=np.array(P);out+=(z0@P.transpose(2,0,1).reshape(n,-1)).reshape(N,L,n)
 return out

def recur(A,B,U,z0=None):
 vals,V=np.linalg.eig(A)
 if np.linalg.cond(V)>1e6:return recur_polynomial(A,B,U,z0)
 inv=np.linalg.inv(V);W=U@(inv@B).T
 for j,v in enumerate(vals):
  W[:,:,j]=lfilter([1.],[1.,-v],W[:,:,j],axis=1)
  if z0 is not None:W[:,:,j]+=(z0@inv.T)[:,j,None]*(v**np.arange(1,U.shape[1]+1))[None,:]
 ans=W@V.T
 if np.iscomplexobj(ans) and np.max(abs(ans.imag))>1e-7:raise FloatingPointError('complex recurrence residual')
 return ans.real

def estep_reference(y,p,stats=True):
 if y.ndim==2:y=y[:,:,None]
 N,T,m=y.shape;A=p['A'];n=len(A);H,K,PP,PF,PS,SI,LD,J,cut=covariance(p,T);mf=np.zeros((N,T,n));state=np.zeros((N,n))
 for t in range(cut):
  if t:state=state@A.T
  state=state+(y[:,t]-state@H.T)@K[t].T;mf[:,t]=state
 if cut<T:
  if p['oracle']:
   mf[:,cut:,0:2]=y[:,cut:,0:2]
   if n==4:mf[:,cut:,3]=y[:,cut-1:T-1,0 if p['kind'] in ['G2','G3'] else 1]
   gain=K[cut,2,2];aa=(1-gain)*A[2,2];v=(1-gain)*(y[:,cut-1:T-1,0]*A[2,0]+y[:,cut-1:T-1,1]*A[2,1])+gain*y[:,cut:,2];mf[:,cut:,2]=lfilter([1.],[1.,-aa],v,axis=1)+mf[:,cut-1,2,None]*(aa**np.arange(1,T-cut+1))[None,:]
  else:mf[:,cut:]=recur((np.eye(n)-K[cut]@H)@A,K[cut],y[:,cut:],mf[:,cut-1])
 prior=np.empty_like(mf);prior[:,0]=0.;prior[:,1:]=mf[:,:-1]@A.T;innov=y-prior@H.T;ll=-.5*(np.einsum('nti,tij,ntj->nt',innov,SI,innov)+LD[None,:]+m*np.log(2*np.pi));loglik=float(ll.sum())
 if not stats:return dict(loglik=loglik,trajectory_ll=ll[:,100:].sum(1),filtered=mf,cov_filter=PF)
 ms=mf.copy()
 if cut<T-1:
  jj=J[cut]
  if p['oracle']:
   j=jj[2,2];u=(mf[:,cut:,2]-j*(mf[:,cut:]@A[2]))[:,::-1].copy();u[:,0]=mf[:,-1,2];ms[:,cut:,2]=lfilter([1.],[1.,-j],u,axis=1)[:,::-1]
  else:
   u=(mf[:,cut:]@(np.eye(n)-jj@A).T)[:,::-1].copy();u[:,0]=mf[:,-1];ms[:,cut:]=recur(jj,np.eye(n),u)[:,::-1]
 for t in range(min(cut,T-1)-1,-1,-1):ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J[t].T
 z0=ms[:,:-1].reshape(-1,n);z1=ms[:,1:].reshape(-1,n);S00=z0.T@z0+N*PS[:-1].sum(0);S11=z1.T@z1+N*PS[1:].sum(0);S10=z1.T@z0+N*np.einsum('tij,tkj->tik',PS[1:],J).sum(0);P0=ms[:,0].T@ms[:,0]/N+PS[0];obs_res=y-ms@H.T;RR=np.einsum('nti,ntj->ij',obs_res,obs_res)/(N*T)+np.einsum('ai,tij,bj->tab',H,PS,H).mean(0)
 return dict(loglik=loglik,smoothed=ms,filtered=mf,S00=S00,S11=S11,S10=S10,P0=P0,obs_var=RR,count=N*(T-1),trajectory_ll=ll[:,100:].sum(1),cov_smooth=PS,cov_filter=PF)

def estep(y,p,stats=True):
 from native_mean import available,means
 if not available:return estep_reference(y,p,stats)
 if y.ndim==2:y=y[:,:,None]
 N,T,m=y.shape;A=p['A'];n=len(A);H,K,PP,PF,PS,SI,LD,J,cut=covariance(p,T);mf,ms=means(y,A,H,K,J,stats);prior=np.empty_like(mf);prior[:,0]=0.;prior[:,1:]=mf[:,:-1]@A.T;innov=y-prior@H.T;ll=-.5*(np.einsum('nti,tij,ntj->nt',innov,SI,innov)+LD[None,:]+m*np.log(2*np.pi));loglik=float(ll.sum())
 if not stats:return dict(loglik=loglik,trajectory_ll=ll[:,100:].sum(1),filtered=mf,cov_filter=PF)
 z0=ms[:,:-1].reshape(-1,n);z1=ms[:,1:].reshape(-1,n);S00=z0.T@z0+N*PS[:-1].sum(0);S11=z1.T@z1+N*PS[1:].sum(0);S10=z1.T@z0+N*np.einsum('tij,tkj->tik',PS[1:],J).sum(0);P0=ms[:,0].T@ms[:,0]/N+PS[0];obs_res=y-ms@H.T;RR=np.einsum('nti,ntj->ij',obs_res,obs_res)/(N*T)+np.einsum('ai,tij,bj->tab',H,PS,H).mean(0)
 return dict(loglik=loglik,smoothed=ms,filtered=mf,S00=S00,S11=S11,S10=S10,P0=P0,obs_var=RR,count=N*(T-1),trajectory_ll=ll[:,100:].sum(1),cov_smooth=PS,cov_filter=PF)

def mstep(e,p):
 A=p['A'].copy();n,mask,buf=setup(p['kind'],p['delay']);phys=2 if p['kind']=='single' else 3
 for i,cols in enumerate(mask):
  V=e['S00'][np.ix_(cols,cols)];s=e['S10'][i,cols];lo=np.zeros(len(cols));hi=np.array([.98 if i==j else 3. for j in cols]);coef=np.linalg.solve(V+np.eye(len(cols))*1e-10,s)
  if np.any(coef<lo) or np.any(coef>hi):
   C=np.linalg.cholesky(V+np.eye(len(cols))*1e-10);coef=lsq_linear(C.T,np.linalg.solve(C,s),bounds=(lo,hi),tol=1e-10,max_iter=100).x
  A[i]=0.;A[i,cols]=coef
 if buf is not None:A[3]=0.;A[3,buf]=1.
 E=e['S11']-A@e['S10'].T-e['S10']@A.T+A@e['S00']@A.T;q=np.diag(p['Q']).copy();upd=range(phys) if p['oracle'] else [phys-1]
 for i in upd:q[i]=float(np.clip(E[i,i]/e['count'],1e-5,5.))
 r=float(np.clip(e['obs_var'][-1,-1],1e-5,5.));P0=solve_discrete_lyapunov(A,np.diag(q))
 return {**p,'A':A,'Q':np.diag(q),'r':r,'P0':P0}

def fit(y,kind,delay,seed,oracle=False,maxiter=40):
 p=initialize(kind,delay,seed,oracle);trace=[];previous=None;last=None;reason='iteration_cap';total=y.shape[0]*y.shape[1]*(y.shape[2] if y.ndim==3 else 1)
 for it in range(maxiter):
  e=estep(y,p);ll=e['loglik'];trace.append(float(ll))
  if last is not None and ll<last-1e-6*total:
   p=previous;trace.pop();reason='monotonicity_safeguard';break
  if last is not None and it>=8 and (ll-last)/total<3e-5:reason='likelihood_tolerance';break
  if it==maxiter-1:break
  previous=p;last=ll;p=mstep(e,p)
 return p,dict(trace=trace,iterations=len(trace),stop=reason)

def serialize(p):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()}
def restore(p):return {**p,**{k:np.array(p[k],dtype=float) for k in ['A','Q','P0']}}
def mirror(p):
 ix=[1,0,2]+([3] if len(p['A'])==4 else []);P=np.eye(len(ix))[ix];return {**p,'A':P@p['A']@P.T,'Q':P@p['Q']@P.T,'P0':P@p['P0']@P.T,'kind':('G4' if p['kind']=='G2' else 'G2')}
