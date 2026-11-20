"""Frozen original L1 encoder. No original truth readouts are loaded."""
import numpy as np,torch
from torch import nn
from settings import ROOT
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(20,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2));self.head=nn.Linear(3,1)
 def forward(self,x):return self.encoder(x)
def encode(x,seed):
 torch.set_num_threads(1);ck=torch.load(ROOT/f'legacy/seed{seed}.pt',weights_only=False,map_location='cpu');m=Net();m.load_state_dict(ck['state']);m.eval();s=(x-ck['X_mean'])/ck['X_sd'];v=np.lib.stride_tricks.sliding_window_view(s,20,axis=1).reshape(-1,20);out=[]
 with torch.no_grad():
  for k in range(0,len(v),8192):out.append(m(torch.tensor(v[k:k+8192].copy(),dtype=torch.float32)).numpy())
 return np.concatenate(out).reshape(len(x),x.shape[1]-19,2)

def linear_head(features,targets,postcov):
 f=features.reshape(-1,features.shape[-1]);y=targets.reshape(-1,2);mu=f.mean(0);sd=np.maximum(f.std(0),1e-8);D=np.c_[np.ones(len(f)),(f-mu)/sd];coef=np.linalg.solve(D.T@D+np.diag([0.]+[1e-3]*f.shape[-1]),D.T@y);res=y-D@coef;P=res.T@res/len(res)+postcov.mean(0);return dict(mean=mu.tolist(),sd=sd.tolist(),coef=coef.tolist(),covariance=P.tolist(),source='X-only learned-state targets plus learned conditional covariance; no true Y/Z')
def apply(features,h):
 shape=features.shape[:-1];f=features.reshape(-1,features.shape[-1]);D=np.c_[np.ones(len(f)),(f-np.array(h['mean']))/np.array(h['sd'])];return (D@np.array(h['coef'])).reshape(*shape,2)
