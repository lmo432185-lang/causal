"""Numerical/prefix checks on independent synthetic development noise only."""
import json,time,numpy as np
from settings import ROOT
from model import train,features,FIXED,RANDOM
from core import predict

def main():
 rng=np.random.default_rng(819910);x=rng.normal(size=(60,500));v=rng.normal(size=(20,500));m=train(x,v);f,y=features(v,m);changed=v.copy();changed[:,181:]+=19.;g,z=features(changed,m);err=0
 for method in ['recursive','direct_full','direct_fixed','direct_random']:
  err=max(err,float(np.max(abs(predict(m['heads'][method],f[method][:,:141])-predict(m['heads'][method],g[method][:,:141])))))
 assert err<1e-10;assert np.max(abs(FIXED.T@FIXED-np.eye(21)))<1e-10
 (ROOT/'DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),unrelated_noise_only=True,training_trajectories=60,max_prefix_error=err,passed=True,total_cpu=m['total_cpu'],timing=m['timing']),indent=2));print('DEV PASSED',m['total_cpu'],flush=True)
if __name__=='__main__':main()
