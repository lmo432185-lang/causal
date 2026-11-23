"""Secondary conditional-on-fitted-model calibration; not physical Y/Z calibration."""
import json,time
import numpy as np,pandas as pd
from scipy.stats import chi2,norm
from settings import *
from ssm import unpack
import native

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  if j['seed']!=3011:continue
  bundle=unpack(json.loads((OUT/j['case']/'model_100.json').read_text()))
  for bank in ['direct','recursive','witness']:
   m=bundle[bank]['models'][2];F=m['F'];Q=m['Q'];C=m['C'];rng=np.random.default_rng(83100+j['index']+len(bank));N=200;z=rng.multivariate_normal(np.zeros(2),m['P'],size=N);x=np.zeros((N,T));truth=np.empty((N,T,2));L=np.linalg.cholesky(Q)
   for t in range(T):
    if t:z=z@F.T+rng.normal(size=z.shape)@L.T
    truth[:,t]=z;x[:,t]=(m['a']*x[:,t-1] if t else 0)+z@C+rng.normal(size=N)*np.sqrt(m['R'])
   p=native.run(x,m);res=truth[:,40:]-p['mean'][:,40:];V=p['cov'][40:];sd=np.sqrt(np.diagonal(V,axis1=1,axis2=2));mah=np.einsum('nti,tij,ntj->nt',res,np.linalg.inv(V),res)
   rows.append(dict(case=j['case'],bank=bank,conditional_model_only=True,trajectories=N,marginal_cover90=float(np.mean(abs(res/sd)<=norm.ppf(.95))),joint_cover90=float(np.mean(mah<=chi2.ppf(.9,2)))))
 pd.DataFrame(rows).to_csv(OUT/'internal_calibration.csv',index=False);(ROOT/'INTERNAL_CALIBRATION_DONE.json').write_text(json.dumps(dict(time=time.time(),own_model_conditional_calibration=True,physical_hidden_calibration_claimed=False,rows=len(rows)),indent=2));print('Internal calibration done',len(rows))
if __name__=='__main__':main()
