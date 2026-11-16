"""Generic X-only reduced-rank predictive state: candidate dimension not supplied by truth."""
import numpy as np
HORIZONS=[1,2,4,8]
def data(x):
 h=np.lib.stride_tricks.sliding_window_view(x,20,axis=1)[:,:-8].copy().reshape(-1,20);targets=np.stack([x[:,19+k:500-8+k].reshape(-1) for k in HORIZONS],1);return h,targets

def fit_x(train,validation):
 # Fit all ranks on train X only; select solely using future validation X.
 mu=train.mean();sd=train.std();tr=(train-mu)/sd;va=(validation-mu)/sd;h,y=data(tr);vh,vy=data(va);c=np.c_[np.ones(len(h)),h[:,-1]];vc=np.c_[np.ones(len(vh)),vh[:,-1]];bx=np.linalg.lstsq(c,h,rcond=None)[0];by=np.linalg.lstsq(c,y,rcond=None)[0];xr=h-c@bx;yr=y-c@by;S=xr.T@xr/len(h);ev,V=np.linalg.eigh(S);W=(V*(1/np.sqrt(np.maximum(ev,1e-8))))@V.T;wh=xr@W;cross=wh.T@yr/len(h);U,s,VT=np.linalg.svd(cross,full_matrices=False);E=W@U;head=np.diag(s)@VT;vl=[]
 for rank in range(5):
  l=(vh-vc@bx)@E[:,:rank];pred=vc@by+l@head[:rank];loss=((pred-vy)**2).reshape(200,-1,4).mean((1,2));vl.append(float(loss.mean()))
 best=min(vl);rank=next(k for k,v in enumerate(vl) if v<=best+1e-6)
 return dict(mu=float(mu),sd=float(sd),bx=bx.tolist(),by=by.tolist(),encoder=E.tolist(),head=head.tolist(),rank=rank,validation_MSE=vl)
def apply_x(x,m):
 h,y=data((x-m['mu'])/m['sd']);c=np.c_[np.ones(len(h)),h[:,-1]];rank=m['rank'];l=(h-c@np.array(m['bx']))@np.array(m['encoder'])[:,:rank];p=c@np.array(m['by'])+l@np.array(m['head'])[:rank];return l,p*m['sd']+m['mu'],y*m['sd']+m['mu']
