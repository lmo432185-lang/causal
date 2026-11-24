"""Exact Gaussian likelihood MCMC in identifiable canonical coordinates; rho remains prior."""
import ctypes,hashlib,subprocess,numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT
src=ROOT/'bayes_likelihood.cpp';lib=ROOT/('bayes_native_'+hashlib.sha256(src.read_bytes()).hexdigest()[:12]+'.so')
if not lib.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
DLL=ctypes.CDLL(str(lib));DLL.ll2.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*5;DLL.ll2.restype=ctypes.c_double
BOUNDS=np.array([[-4,4],[-10,10],[-14,4],[-14,4],[-14,3]],float);CENTER=np.array([0.,0.,np.log(.5),np.log(.5),np.log(.3)]);SD=np.array([1.5,3.,1.2,1.2,1.2])
def unpack_theta(z):
 a=.9995*np.tanh(z[0]);b=-.9995+(a+.9995)*expit(z[1]);return a,b,*np.exp(z[2:])
def to_theta(m):
 a,b=np.diag(m['F']);k=np.diag(m['P']);return np.clip(np.r_[np.arctanh(np.clip(a/.9995,-.99999,.99999)),logit(np.clip((b+.9995)/(a+.9995),1e-5,1-1e-5)),np.log(k),np.log(m['R'])],BOUNDS[:,0]+1e-4,BOUNDS[:,1]-1e-4)
def make_model(z,scale):
 a,b,k1,k2,R=unpack_theta(z);F=np.diag([a,b]);Q=np.diag([k1*(1-a*a),k2*(1-b*b)]);return dict(F=F,Q=Q,C=np.ones(2),P=np.diag([k1,k2]),R=R,a=0.,scale=scale,kind='factor2',state_dimension=2,factor_count=2)
def likelihood(x,z):
 a,b,k1,k2,R=unpack_theta(z);return DLL.ll2(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(x),x.shape[1],a,b,k1,k2,R)
def posterior(x,z):
 if np.any(z<BOUNDS[:,0]) or np.any(z>BOUNDS[:,1]):return -np.inf
 return likelihood(x,z)-.5*np.sum(((z-CENTER)/SD)**2)
def diagnostics(chain):
 n=chain.shape[1];W=np.var(chain,axis=1,ddof=1).mean(0);B=n*np.var(chain.mean(1),axis=0,ddof=1);rhat=np.where(W>1e-20,np.sqrt(((n-1)/n*W+B/n)/np.maximum(W,1e-20)),np.inf);ess=[]
 for k in range(5):
  u=chain[:,:,k]-chain[:,:,k].mean(1)[:,None];ff=np.fft.rfft(u,n=2*n,axis=1);g=np.fft.irfft(ff*ff.conj(),n=2*n,axis=1)[:,:n].mean(0)/np.arange(n,0,-1);rho=g/np.maximum(g[0],1e-20);pairs=rho[1:-1:2]+rho[2::2];end=np.where(pairs<=0)[0];cut=int(end[0]) if len(end) else len(pairs);ess.append(0. if g[0]<=1e-20 else float(min(2*n,2*n/max(1,1+2*pairs[:cut].sum()))))
 return rhat.tolist(),ess

def sample(x,initial,seed,warm=1000,draws=3000):
 x=np.ascontiguousarray(x,dtype=np.float64);z0=to_theta(initial);opt=minimize(lambda z:-posterior(x,z),z0,method='L-BFGS-B',bounds=BOUNDS.tolist(),options={'maxiter':150,'ftol':1e-11});mode=opt.x;h=1e-3;H=np.empty((5,5));f=lambda z:-posterior(x,z)
 for i in range(5):
  ei=np.eye(5)[i]*h
  for j in range(5):
   ej=np.eye(5)[j]*h;H[i,j]=(f(mode+ei+ej)-f(mode+ei-ej)-f(mode-ei+ej)+f(mode-ei-ej))/(4*h*h)
 if not np.isfinite(H).all():H=np.eye(5)*100
 v,U=np.linalg.eigh((H+H.T)/2);cov=(U/np.maximum(v,1e-3))@U.T;cap=np.array([.15,.5,.5,.5,.3]);factor=np.minimum(1,cap/np.sqrt(np.maximum(np.diag(cov),1e-12)));cov=cov*factor[:,None]*factor[None];cov+=1e-8*np.eye(5);chains=[];rates=[]
 for c in range(2):
  rng=np.random.default_rng(seed+c*9917);z=mode.copy() if c==0 else np.clip(mode+rng.multivariate_normal(np.zeros(5),cov)*.2,BOUNDS[:,0]+1e-5,BOUNDS[:,1]-1e-5);lp=posterior(x,z);history=[];saved=[];accepted=0;window=0;step=2.38**2/5;proposal=cov.copy();L=np.linalg.cholesky(proposal*step)
  for it in range(warm+draws):
   zp=z+L@rng.normal(size=5);p=posterior(x,zp)
   if np.log(rng.random())<p-lp:z=zp;lp=p;accepted+=1;window+=1
   if it<warm:
    history.append(z.copy())
    if (it+1)%50==0:
     step*=np.exp(np.clip(window/50-.234,-.2,.2));window=0
     if it>=199:proposal=.9*np.cov(np.array(history).T)+.1*cov+1e-8*np.eye(5)
     L=np.linalg.cholesky(proposal*step)
   else:saved.append(z.copy())
  chains.append(saved);rates.append(accepted/(warm+draws))
 chain=np.array(chains);rhat,ess=diagnostics(chain);return dict(chain=chain,mode=mode,rhat=rhat,ESS=ess,acceptance=rates,optimizer_success=bool(opt.success),optimizer_message=str(opt.message),converged=max(rhat)<=1.1 and min(ess)>=100 and min(rates)>=.05 and max(rates)<=.8)
