"""Fixed pre-registered C-grid seed 4011 only. No choice of main results follows this audit."""
import json,gzip,time,numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from producer import generate,matrices,SYSTEMS
from model import unpack
from probability import nodes,correlated_alias,physical_scores,TIMES
from state_audit import qphysical

def evaluate(x,y,theta,nrho):
 rho,wr=nodes(nrho);ms=[correlated_alias(m,r) for m in theta for r in rho];prior=np.array([-np.log(len(theta))+np.log(w) for m in theta for w in wr]);mu,cov,w,_=qphysical(x,ms,prior);P=len(x)*len(TIMES);sc=physical_scores(mu.reshape(P,len(ms),2),cov.reshape(P,len(ms),2,2),w.reshape(P,len(ms)),y[:,TIMES].reshape(P,2));out={k:float(v.mean()) for k,v in sc.items()};out['rho_posterior_error']=float(np.max(abs(w.reshape(len(x),len(TIMES),len(theta),nrho).sum(2)-wr)));return out

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[]
 for index in range(20,24):
  case=SYSTEMS[index]['id']+'_4011'
  with gzip.open(OUT/'bayes'/(case+'.json.gz'),'rt') as f:b=unpack(json.load(f))
  for batch,stage in [('A',11),('B',12)]:
   x,y=generate(index,4011,stage,NTEST,truth=True);base=evaluate(x,y,b['models8'],17)
   for name,theta,nrho in [('theta16',b['models16'],17),('rho33',b['models8'],33)]:
    z=evaluate(x,y,theta,nrho);cover=abs(z['cover90']-base['cover90']);width=abs(z['width90']/base['width90']-1);nll=abs(z['joint_NLL']-base['joint_NLL']);rows.append(dict(condition=SYSTEMS[index]['id'],seed=4011,batch=batch,diagnostic=name,cover90_difference=cover,width90_relative_difference=width,joint_NLL_difference=nll,rho_prior_error=z['rho_posterior_error'],stable=cover<=.02 and width<=.1 and nll<=.05))
 pd.DataFrame(rows).to_csv(OUT/'integration_sensitivity.csv',index=False);(ROOT/'SENSITIVITY_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),tests_not_retuned=True,comparisons=len(rows),all_stable=all(x['stable'] for x in rows)),indent=2));print('Fixed integration sensitivity audit complete',flush=True)
if __name__=='__main__':main()
