import numpy as np
from config import HORIZONS

def generate(g,seed):
 rng=np.random.default_rng(seed);n=1000;dim=2 if g['kind']=='direct' else 3;s=np.zeros((n,dim));states=[]
 for t in range(700):
  if dim==2:A=np.array([[g['a'],0.],[g['beta'],g['b']]])
  else:A=np.array([[g['a'],0,0],[g['alpha'],g['c'],0],[g['beta1'],g['beta'],g['b']]])
  q=np.array([g['qy'],g['qx']]) if dim==2 else np.array([g['qy'],g['qz'],g['qx']]);s=s@A.T+rng.normal(size=s.shape)*np.sqrt(q)
  if t>=200:states.append(s.copy())
 a=np.stack(states,axis=1);return a[:,:,-1],a

def hist(x,start=19,stop=492,p=20):return np.stack([x[:,t-p+1:t+1] for t in range(start,stop)],1)
def olsfit(x,y):
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);z=np.c_[np.ones(len(x)),(x-mu)/sd];coef=np.linalg.pinv(z.T@z,rcond=1e-10)@(z.T@y);return dict(mean=mu.tolist(),sd=sd.tolist(),coef=coef.tolist())
def olsapply(x,m):return np.c_[np.ones(len(x)),(x-np.array(m['mean']))/np.array(m['sd'])]@np.array(m['coef'])

def fit_predictive(train,val):
 mu=train.mean();sd=train.std();x=(train-mu)/sd;v=(val-mu)/sd;h=hist(x).reshape(-1,20);vh=hist(v).reshape(-1,20);y=np.stack([x[:,19+k:492+k].reshape(-1) for k in HORIZONS],1);vy=np.stack([v[:,19+k:492+k].reshape(-1) for k in HORIZONS],1);c=np.c_[np.ones(len(h)),h[:,-1]];vc=np.c_[np.ones(len(vh)),vh[:,-1]];bx=np.linalg.lstsq(c,h,rcond=None)[0];by=np.linalg.lstsq(c,y,rcond=None)[0];xr=h-c@bx;yr=y-c@by;ev,V=np.linalg.eigh(xr.T@xr/len(xr));W=(V*(1/np.sqrt(np.maximum(ev,1e-8))))@V.T;U,s,VT=np.linalg.svd((xr@W).T@yr/len(xr),full_matrices=False);E=W@U;head=np.diag(s)@VT;vl=[]
 for rank in range(5):vl.append(float(np.mean((vc@by+(vh-vc@bx)@E[:,:rank]@head[:rank]-vy)**2)))
 rank=next(i for i,q in enumerate(vl) if q<=min(vl)+1e-6);return dict(mu=float(mu),sd=float(sd),bx=bx.tolist(),by=by.tolist(),E=E.tolist(),head=head.tolist(),rank=rank,validation_MSE=vl)

def encode(x,m,rank=None):
 k=m['rank'] if rank is None else rank;x=(x-m['mu'])/m['sd'];h=hist(x,19,500).reshape(-1,20);c=np.c_[np.ones(len(h)),h[:,-1]];L=(h-c@np.array(m['bx']))@np.array(m['E'])[:,:k];return L.reshape(len(x),481,k)
def forecast(x,m,rank=None):
 k=m['rank'] if rank is None else rank;l=encode(x,m,k)[:,:473].reshape(len(x)*473,k);c=np.c_[np.ones(len(l)),((x[:,19:492]-m['mu'])/m['sd']).reshape(-1)];return (c@np.array(m['by'])+l@np.array(m['head'])[:k])*m['sd']+m['mu']

def forecast_design(x,L,baseline):
 h=hist(x).reshape(-1,20);base=h[:,-1:] if baseline=='current_X' else h;lat=L[:,:473].reshape(len(h),L.shape[-1]);y=np.stack([x[:,19+k:492+k].reshape(-1) for k in HORIZONS],1);return base,lat,y

def direction_design(x,S,target,source,p,covered=False):
 # S[n,t,node] indexed at physical t=19..499. Forecast physical t+1.
 k=S.shape[-1];start=19+p-1;stop=499;ns=stop-start;lagged=np.concatenate([S[:,start-19-l:stop-19-l,:] for l in range(p)],2).reshape(-1,p*k);y=S[:,start-19+1:stop-19+1,target].reshape(-1,1);drop=[l*k+source for l in range(p)];keep=[j for j in range(p*k) if j not in drop];full=lagged;reduced=lagged[:,keep]
 if covered:
  hh=hist(x,start,stop,20+p-1).reshape(-1,20+p-1);full=np.c_[hh,full];reduced=np.c_[hh,reduced]
 return full,reduced,y,ns
