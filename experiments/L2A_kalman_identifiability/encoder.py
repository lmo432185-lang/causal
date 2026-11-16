import torch
from torch import nn
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(20,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2));self.head=nn.Linear(3,1)
 def forward(self,x):
  l=self.encoder(x);return self.head(torch.cat([x[:,-1:],l],1)),l

def encode(m,x):
 with torch.no_grad():return torch.cat([m.encoder(v) for v in torch.from_numpy(x.astype('float32')).split(8192)]).numpy()
def apply_map(x,params):
 import numpy as np
 mu=np.array(params['mean'],dtype=x.dtype);sd=np.array(params['sd'],dtype=x.dtype);d=np.c_[np.ones(len(x)),(x-mu)/sd];return d@np.array(params['coef'])
