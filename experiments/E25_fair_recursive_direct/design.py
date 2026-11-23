"""Analytic generator design only; never imports learned encoders or simulated data."""
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from pathlib import Path
ROOT=Path(__file__).parent

def white(C):
 w,v=np.linalg.eigh(C);return (v/np.sqrt(np.maximum(w,1e-12)))@v.T

def information(A,q):
 A=np.array(A);p=solve_discrete_lyapunov(A,np.diag(q));g=np.array([(np.linalg.matrix_power(A,k)@p)[-1,-1] for k in range(51)]);past=np.arange(1,41);f=np.arange(1,11)
 cp=g[abs(past[:,None]-past)]-np.outer(g[past],g[past])/g[0];cf=g[abs(f[:,None]-f)]-np.outer(g[f],g[f])/g[0];cross=g[past[:,None]+f]-np.outer(g[past],g[f])/g[0];s=np.linalg.svd(white(cp)@cross@white(cf),compute_uv=False);return (-.5*np.log1p(-np.minimum(s*s,1-1e-12))/10).tolist()

def main():
 systems=[dict(id='S0',family='S0',strength='none',A=[[.7]],Q=[1.]),dict(id='S1',family='S1',strength='single',A=[[.97,0],[1.,0]],Q=[.25,4.])]
 for family in ['S2','S3','S4']:
  candidates=[]
  for speed in [-.97,-.9,-.7,-.3,.3,.7,.9,.97]:
   for noise in np.geomspace(.05,100,55):
    if family=='S2':A=[[.97,0,0],[.8,speed,0],[0,1.,0]];q=[.25,.25,float(noise)]
    elif family=='S3':A=[[.97,0,0],[0,speed,0],[1.,1.,0]];q=[.25,.25,float(noise)]
    else:A=[[.8,0,0,0],[.4,.97,0,0],[.4,0,speed,0],[0,1.,1.,0]];q=[.25,.25,.25,float(noise)]
    info=information(A,q);candidates.append(dict(A=A,Q=q,oracle_information=info,speed=speed))
  for label,target in [('weak',.001),('medium',.005),('strong',.015)]:
   m=min(candidates,key=lambda d:abs(np.log(max(d['oracle_information'][1],1e-12)/target)));systems.append(dict(id=family+'_'+label,family=family,strength=label,target_second=target,**m))
 for m in systems:
  if 'oracle_information' not in m:m['oracle_information']=information(m['A'],m['Q'])
 (ROOT/'SYSTEMS.json').write_text(json.dumps(systems,indent=2));print('\n'.join(f"{m['id']} speed={m.get('speed')} noise={m['Q'][-1]:.4f} info={m['oracle_information'][:3]}" for m in systems))
if __name__=='__main__':main()
