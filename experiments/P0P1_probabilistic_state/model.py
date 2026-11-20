"""Unknown-parameter chain, exact Gaussian filtering, X-only composite-likelihood fitting."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov,toeplitz,cho_factor,cho_solve
from scipy.optimize import minimize
from native_mean import available,means
from settings import BLOCK,MAXITER
BOUNDS=[(0.,.985)]*3+[(.005,3.),(.005,3.),(1e-5,5.),(1e-5,5.)]
NFFT=4096
W=2*np.pi*np.arange(NFFT)/NFFT;COS=np.cos(W)

def plant(theta,delay=1):
 a,b,c,u,v,qx,r=theta;d=int(delay);n=3+2*(d-1);A=np.zeros((n,n));A[0,0]=a;A[1,1]=b;A[2,2]=c;A[1,0 if d==1 else d+1]=u;A[2,1 if d==1 else 2*d]=v
 if d>1:
  A[3,0]=1.;A[d+2,1]=1.
  for k in range(1,d-1):A[3+k,2+k]=1.;A[d+2+k,d+1+k]=1.
 Q=np.diag([1.,1.,qx]+[0.]*(n-3));P=solve_discrete_lyapunov(A,Q);return A,Q,float(r),P

def spectrum(theta,grad=False):
 a,b,c,u,v,qx,r=theta;Da=1+a*a-2*a*COS;Db=1+b*b-2*b*COS;Dc=1+c*c-2*c*COS
 tx=qx/Dc;tz=v*v/(Db*Dc);ty=v*v*u*u/(Da*Db*Dc);S=tx+tz+ty+r
 if not grad:return S
 G=np.stack([-ty*(2*a-2*COS)/Da,-(tz+ty)*(2*b-2*COS)/Db,-(tx+tz+ty)*(2*c-2*COS)/Dc,2*ty/u,2*(tz+ty)/v,1/Dc,np.ones(NFFT)])
 return S,G

def block_statistics(x):
 # Exact stationary marginal covariance of each nonoverlapping block; dependence between blocks is not counted as independent replication in uncertainty estimates.
 nb=x.shape[1]//BLOCK;z=x[:,:nb*BLOCK].reshape(-1,BLOCK);return z.T@z/len(z)

def objective(theta,Sdata,grad=True):
 S,G=spectrum(theta,True);g=np.fft.ifft(S).real[:BLOCK];C=toeplitz(g);cf=cho_factor(C,lower=True,check_finite=False);iv=cho_solve(cf,np.eye(BLOCK),check_finite=False);value=.5*(2*np.log(np.diag(cf[0])).sum()+np.sum(iv*Sdata))
 D=iv-iv@Sdata@iv;dg=np.fft.ifft(G,axis=1).real[:,:BLOCK];lags=np.abs(np.arange(BLOCK)[:,None]-np.arange(BLOCK)[None,:]);der=.5*np.einsum('ij,kij->k',D,dg[:,lags])
 return float(value),der

def fit(Sdata,seed,maxiter=MAXITER,initial=None):
 rng=np.random.default_rng(seed);init=np.array([rng.uniform(.2,.92),rng.uniform(.2,.92),rng.uniform(.2,.85),rng.uniform(.15,.9),rng.uniform(.12,.7),rng.uniform(.1,.7),rng.uniform(.02,.25)]) if initial is None else np.array(initial).copy();trace=[]
 def cb(t):trace.append(float(objective(t,Sdata)[0]))
 z=minimize(objective,init,args=(Sdata,),method='L-BFGS-B',jac=True,bounds=BOUNDS,callback=cb,options=dict(maxiter=maxiter,ftol=1e-10,gtol=1e-6,maxls=30));return z.x,dict(success=bool(z.success),message=str(z.message),nit=int(z.nit),nfev=int(z.nfev),objective=float(z.fun),raw_gradient_max=float(np.max(abs(z.jac))),trace=trace,initial=init.tolist())

def filter_cov(A,Q,R,P0,T,smooth=True):
 n=len(A);H=np.eye(n)[2:3];P=P0.copy();K=np.zeros((T,n,1));PP=np.empty((T,n,n));PF=np.empty_like(PP);S=np.empty(T)
 for t in range(T):
  PP[t]=P;S[t]=P[2,2]+R;K[t,:,0]=P[:,2]/S[t];F=P-np.outer(P[:,2],P[2])/S[t];PF[t]=(F+F.T)*.5;P=A@PF[t]@A.T+Q
 J=np.zeros((T-1,n,n));PS=PF.copy()
 if smooth:
  for t in range(T-1):J[t]=PF[t]@A.T@np.linalg.pinv(PP[t+1],rcond=1e-12,hermitian=True)
  for t in range(T-2,-1,-1):PS[t]=PF[t]+J[t]@(PS[t+1]-PP[t+1])@J[t].T;PS[t]=(PS[t]+PS[t].T)*.5
 return H,K,PP,PF,S,J,PS

def infer(x,params,smooth=False):
 A,Q,R,P0=params;N,T=x.shape;n=len(A);H,K,PP,PF,S,J,PS=filter_cov(A,Q,R,P0,T,smooth)
 if available:mf,ms=means(x[:,:,None],A,H,K,J,smooth)
 else:
  mf=np.zeros((N,T,n));z=np.zeros((N,n))
  for t in range(T):
   if t:z=z@A.T
   z+=(x[:,t]-z[:,2])[:,None]*K[t,:,0];mf[:,t]=z
  ms=mf.copy() if smooth else None
  if smooth:
   for t in range(T-2,-1,-1):ms[:,t]=mf[:,t]+(ms[:,t+1]-mf[:,t]@A.T)@J[t].T
 pred=np.zeros_like(x);pred[:,1:]=(mf[:,:-1]@A.T)[:,:,2];innov=x-pred;ll=-.5*(innov**2/S[None,:]+np.log(2*np.pi*S)[None,:]);return dict(mean=mf,cov=PF,smoothed=ms,cov_smooth=PS,forecast=pred,forecast_var=S,trajectory_ll=ll[:,50:].sum(1),innovations=innov)

def serialize(p):return dict(A=p[0].tolist(),Q=p[1].tolist(),R=p[2],P0=p[3].tolist())
def restore(p):return np.array(p['A']),np.array(p['Q']),float(p['R']),np.array(p['P0'])
