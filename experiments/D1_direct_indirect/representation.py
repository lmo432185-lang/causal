"""Experiment1-style predictive/dynamic latent encoder; no hidden labels in fitting."""
import numpy as np,torch
from torch import nn
from config import *
torch.set_num_threads(2)
def net(a,b):return nn.Sequential(nn.Linear(a,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,b))
class Latent(nn.Module):
 def __init__(self):
  super().__init__();self.enc=net(EMBED,2);self.obs=net(3,3);self.dyn=net(3,2)
 def forward(self,h,next_h):
  z=self.enc(h);zn=self.enc(next_h);s=torch.cat([h[:,:1],z],1);return self.obs(s),self.dyn(s)+z,z,zn

def sample_windows(x,mu,sd,seed,count=50000):
 rng=np.random.default_rng(seed);rows=rng.integers(0,len(x),count);ids=rng.integers(EMBED-1,x.shape[1]-max(HORIZONS),count);v=(x-mu)/sd;h=v[rows[:,None],ids[:,None]-np.arange(EMBED)[None,:]];nh=v[rows[:,None],ids[:,None]+1-np.arange(EMBED)[None,:]];y=np.stack([v[rows,ids+j]-v[rows,ids] for j in HORIZONS],1);return [torch.tensor(q,dtype=torch.float32) for q in [h,nh,y]]

def fit(x,xval,seed,mu,sd):
 torch.manual_seed(seed);tr=sample_windows(x,mu,sd,seed);va=sample_windows(xval,mu,sd,seed+100,10000);model=Latent();opt=torch.optim.Adam(model.parameters(),lr=.002);best=np.inf;checkpoint=None
 for step in range(NET_STEPS):
  idx=torch.randint(len(tr[0]),(512,));p,d,z,zn=model(tr[0][idx],tr[1][idx]);loss=((p-tr[2][idx])**2).mean()+.2*((d-zn)**2).mean()+.02*(z.mean(0)**2).mean()+.02*((z.std(0)-1)**2).mean();opt.zero_grad();loss.backward();opt.step()
  if step%100==99:
   with torch.no_grad():err=float(((model(va[0],va[1])[0]-va[2])**2).mean())
   if err<best:best=err;checkpoint={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(checkpoint);return model.eval(),best

def encode(model,x,mu,sd,ids):
 v=(x-mu)/sd;h=v[:,ids[:,None]-np.arange(EMBED)[None,:]].reshape(-1,EMBED)
 with torch.no_grad():z=model.enc(torch.tensor(h,dtype=torch.float32)).numpy()
 return z.reshape(len(x),len(ids),2)

def diagnostic_readout(model,val,mu,sd):
 # Hidden validation answers used only AFTER model is frozen; not used in training/selection.
 ids=np.arange(EMBED-1,val['x'].shape[1],20);z=encode(model,val['x'],mu,sd,ids);xx=val['x'][:,ids];design=np.c_[np.ones(xx.size),xx.ravel(),z.reshape(-1,2)];base=np.c_[np.ones(xx.size),xx.ravel()];out={}
 for key in ['Y','W']:
  target=val[key][:,ids].ravel();out[key]=np.linalg.lstsq(design,target,rcond=None)[0];out[key+'_base']=np.linalg.lstsq(base,target,rcond=None)[0]
 return out
