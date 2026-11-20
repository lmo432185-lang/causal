"""Gaussian anchored chain, block composite likelihood and exact filtering.
Y/Z unit innovation is a coordinate gauge. hY/hZ link it to observed units;
known source input uses unknown lambda=1/hY to preserve physical input gain1.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov,cho_factor,cho_solve
from scipy.optimize import minimize
from native_mean import available,means
from settings import BLOCK,AUX_STD,INPUT_STD

def plant(theta,delay,kind):
 a,b,c,u,v,qx,r,hY,hZ,lam=theta;d=int(delay);nb=3+2*(d-1);n=nb+int(kind=='E3');A=np.zeros((n,n));A[0,0]=a;A[1,1]=b;A[2,2]=c;A[1,0 if d==1 else d+1]=u;A[2,1 if d==1 else 2*d]=v
 if d>1:
  A[3,0]=1.;A[d+2,1]=1.
  for k in range(1,d-1):A[3+k,2+k]=1.;A[d+2+k,d+1+k]=1.
 Q=np.diag([1.,1.,qx]+[0.]*(nb-3)+([INPUT_STD**2] if kind=='E3' else []))
 if kind=='E3':A[0,-1]=lam
 P=solve_discrete_lyapunov(A,Q);return A,Q,r,P

def derivatives(theta,delay,kind,A,Q,P):
 n=len(A);da=np.zeros((10,n,n));dq=np.zeros_like(da);da[0,0,0]=1;da[1,1,1]=1;da[2,2,2]=1;da[3,1,0 if delay==1 else delay+1]=1;da[4,2,1 if delay==1 else 2*delay]=1;dq[5,2,2]=1
 if kind=='E3':da[9,0,-1]=1
 dp=np.zeros_like(da)
 for k in [0,1,2,3,4,5]+([9] if kind=='E3' else []):dp[k]=solve_discrete_lyapunov(A,da[k]@P@A.T+A@P@da[k].T+dq[k])
 return da,dp

def observations(kind,nb):
 obs=[(2,t) for t in range(BLOCK)]
 if kind=='E3':obs += [(nb,t) for t in range(BLOCK)]
 elif kind in ['E4Y','E4Z']:
  obs += [(0 if kind=='E4Y' else 1,t) for t in [20]]
 elif kind=='E4YZ':obs += [(node,t) for node in [0,1] for t in [20]]
 return np.array(obs,dtype=int)

def covariance(theta,delay,kind,grad=True):
 A,Q,r,P=plant(theta,delay,kind);nb=3+2*(delay-1);obs=observations(kind,nb);nodes=obs[:,0];times=obs[:,1];lag=times[:,None]-times[None,:];pos=lag>=0;ii=np.where(pos,nodes[:,None],nodes[None,:]);jj=np.where(pos,nodes[None,:],nodes[:,None]);lag=abs(lag);G=np.empty((BLOCK,len(A),len(A)));G[0]=P
 if grad:da,dp=derivatives(theta,delay,kind,A,Q,P);DG=np.empty((10,BLOCK,len(A),len(A)));DG[:,0]=dp
 for t in range(1,BLOCK):
  G[t]=A@G[t-1]
  if grad:DG[:,t]=da@G[t-1]+A@DG[:,t-1]
 load=np.ones(len(obs));load[nodes==0]=theta[7];load[nodes==1]=theta[8];C0=G[lag,ii,jj];C=C0*load[:,None]*load[None,:];diag=np.arange(len(obs));isx=nodes==2;isaux=(nodes==0)|(nodes==1);C[diag[isx],diag[isx]]+=r;C[diag[isaux],diag[isaux]]+=AUX_STD**2
 if not grad:return C
 DC=DG[:,lag,ii,jj]*load[:,None]*load[None,:];DC[6,diag[isx],diag[isx]]+=1
 # Boolean addition in NumPy must be cast before adding; diagonal scaling derivative is2.
 for k,node in [(7,0),(8,1)]:DC[k]+=C0*load[:,None]*load[None,:]*((nodes==node).astype(float)[:,None]+(nodes==node).astype(float)[None,:])/theta[k]
 return C,DC

def design(x,kind,aux=None,I=None):
 nb=x.shape[1]//BLOCK;xx=x[:,:nb*BLOCK].reshape(-1,BLOCK);pieces=[xx]
 if kind=='E3':pieces.append(I[:,:nb*BLOCK].reshape(-1,BLOCK))
 elif kind in ['E4Y','E4Z']:
  z=aux['Y' if kind=='E4Y' else 'Z'].reshape(len(x),nb,2);pieces.append(z.reshape(-1,2))
 elif kind=='E4YZ':
  for s in ['Y','Z']:pieces.append(aux[s].reshape(len(x),nb,1).reshape(-1,1))
 return np.concatenate(pieces,1)

def objective(theta,S,delay,kind):
 C,D=covariance(theta,delay,kind);cf=cho_factor(C,lower=True,check_finite=False);iv=cho_solve(cf,np.eye(len(C)),check_finite=False);loss=.5*(2*np.log(np.diag(cf[0])).sum()+np.sum(iv*S));W=iv-iv@S@iv;grad=.5*np.einsum('ij,kij->k',W,D);return float(loss),grad

def encode(theta,kind):
 z=np.array(theta).copy()
 if kind=='E1':z[:3]=[theta[0]/.985,theta[1]/theta[0],theta[2]/theta[1]]
 return z

def decode(z,kind):
 theta=np.array(z).copy();J=np.eye(10)
 if kind=='E1':
  ra,rb,rc=z[:3];theta[:3]=[.985*ra,.985*ra*rb,.985*ra*rb*rc];J[:3,:3]=[[.985,0,0],[.985*rb,.985*ra,0],[.985*rb*rc,.985*ra*rc,.985*ra*rb]]
 return theta,J

def fit(S,kind,delay,seed,anchor_c=None,maxiter=140):
 rng=np.random.default_rng(seed);theta=np.array([rng.uniform(.35,.94),rng.uniform(.25,.84),rng.uniform(.25,.8),rng.uniform(.2,.9),rng.uniform(.15,.55),rng.uniform(.1,.6),rng.uniform(.02,.15),rng.uniform(.75,1.2),rng.uniform(.75,1.2),rng.uniform(.7,1.4)])
 bounds=[(0.,.985)]*3+[(.005,3.),(.005,3.),(1e-5,5.),(1e-5,5.),(.15,3.),(.15,3.),(.05,10.)]
 if kind=='E1':theta[:3]=sorted(theta[:3],reverse=True);bounds[:3]=[(.02,.9999)]*3
 if kind=='E2':theta[2]=anchor_c;bounds[2]=(anchor_c,anchor_c)
 if kind not in ['E4Y','E4YZ']:theta[7]=1.;bounds[7]=(1.,1.)
 if kind not in ['E4Z','E4YZ']:theta[8]=1.;bounds[8]=(1.,1.)
 if kind!='E3':theta[9]=1.;bounds[9]=(1.,1.)
 z0=encode(theta,kind);trace=[]
 def fun(z):
  t,J=decode(z,kind);value,g=objective(t,S,delay,kind);return value,J.T@g
 def cb(z):trace.append(fun(z)[0])
 opt=minimize(fun,z0,method='L-BFGS-B',jac=True,bounds=bounds,callback=cb,options=dict(maxiter=maxiter,ftol=1e-10,gtol=1e-6,maxls=30));t,_=decode(opt.x,kind);return t,dict(success=bool(opt.success),message=str(opt.message),nit=int(opt.nit),nfev=int(opt.nfev),loss=float(opt.fun),trace=trace)

def block_ll(Z,theta,delay,kind):
 C=covariance(theta,delay,kind,False);cf=cho_factor(C,lower=True,check_finite=False);iv=cho_solve(cf,np.eye(len(C)),check_finite=False);return -.5*(np.sum((Z@iv)*Z,axis=1)+2*np.log(np.diag(cf[0])).sum()+len(C)*np.log(2*np.pi))

def infer(x,theta,delay,kind,I=None):
 A,Q,R,P=plant(theta,delay,kind);n=len(A);N,T=x.shape;H=np.eye(n)[2:3];RR=np.array([[R]]);y=x[:,:,None]
 if kind=='E3':H=np.vstack([H,np.eye(n)[-1:]]);RR=np.diag([R,0.]);y=np.stack([x,I],2)
 K=[];PF=[];PP=[];invS=[];ld=[]
 for t in range(T):
  S=H@P@H.T+RR;iv=np.linalg.inv(S);g=P@H.T@iv;F=P-g@H@P;F=(F+F.T)*.5;K.append(g);PF.append(F);PP.append(P.copy());invS.append(iv);ld.append(np.linalg.slogdet(S)[1]);P=A@F@A.T+Q
 K=np.array(K);PF=np.array(PF);J=np.zeros((T-1,n,n));
 if available:mf,_=means(y,A,H,K,J,False)
 else:
  mf=np.zeros((N,T,n));z=np.zeros((N,n))
  for t in range(T):
   if t:z=z@A.T
   z+=(y[:,t]-z@H.T)@K[t].T;mf[:,t]=z
 pred=np.zeros((N,T,H.shape[0]));pred[:,1:]=mf[:,:-1]@A.T@H.T;innov=y-pred;ll=-.5*(np.einsum('nti,tij,ntj->nt',innov,np.array(invS),innov)+np.array(ld)[None,:]+len(H)*np.log(2*np.pi));scale=np.ones(2)
 if kind in ['E4Y','E4YZ']:scale[0]=theta[7]
 if kind in ['E4Z','E4YZ']:scale[1]=theta[8]
 if kind=='E3':scale[0]=1/theta[9]
 # Return declared physical proxy scale; unanchored coordinates retain unit-innovation gauge.
 return dict(mean=mf[:,:,:2]*scale,cov=PF[:,:2,:2]*scale[:,None]*scale[None,:],forecast=pred[:,:,0],forecast_var=np.array([1/s[0,0] if len(H)==1 else (H@p@H.T+RR)[0,0] for s,p in zip(invS,PP)]),trajectory_ll=ll[:,50:].sum(1))
