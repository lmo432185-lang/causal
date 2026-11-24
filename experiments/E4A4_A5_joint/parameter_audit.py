import json,gzip,time,numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from settings import *
from model import unpack
from producer import matrices,SYSTEMS
from bayes import unpack_theta

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists();rows=[];mc=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  meta=SYSTEMS[j['index']]
  if meta['family']=='D' or meta['family']=='A':continue
  F,Q,C,R=matrices(meta);P=solve_discrete_lyapunov(F,Q);truep=np.r_[np.diag(F),P@C,R]
  with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=unpack(json.load(f))
  for bank in ['R','D']:
   m=b['bases'][bank]['canonical_point'];hat=np.r_[np.diag(m['F']),m['P']@C*m['scale']**2,m['R']*m['scale']**2];rows.append(dict(case=j['case'],seed=j['seed'],bank=bank,pole_MAE=float(abs(hat[:2]-truep[:2]).mean()),row_sum_MAE=float(abs(hat[2:4]-truep[2:4]).mean()),R_error=float(abs(hat[4]-truep[4])),converged=m.get('converged',False)))
  path=OUT/'bayes'/(j['case']+'.json.gz')
  if path.exists():
   with gzip.open(path,'rt') as f:bb=unpack(json.load(f))
   values=np.array([unpack_theta(z) for z in bb['chain'].reshape(-1,5)]);values[:,2:]*=bb['scale']**2;lo,hi=np.quantile(values,[.05,.95],axis=0)
   for k,name in enumerate(['pole_a','pole_b','spectral_row1','spectral_row2','R']):mc.append(dict(case=j['case'],seed=j['seed'],parameter=name,mean=float(values[:,k].mean()),lo=float(lo[k]),hi=float(hi[k]),truth=float(truep[k]),covered=bool(lo[k]<=truep[k]<=hi[k]),Rhat=bb['rhat'][k],ESS=bb['ESS'][k],sampler_converged=bb['converged']))
 pd.DataFrame(rows).to_csv(OUT/'canonical_parameter_errors.csv',index=False);pd.DataFrame(mc).to_csv(OUT/'continuous_parameter_audit.csv',index=False);(ROOT/'PARAMETER_AUDIT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),true_parameters_after_tests=True,physical_innovation_covariance_not_treated_identifiable=True),indent=2))
if __name__=='__main__':main()
