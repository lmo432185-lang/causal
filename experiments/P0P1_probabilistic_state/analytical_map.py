"""Supplementary Oracle Riccati map with matched source stationary variance.
No learner is fitted, no test answer is used, no primary gate is changed.
"""
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_are
from settings import ROOT,OUT
from model import plant,infer

def main():
 protocol=dict(time=time.time(),scope='supplementary analytic Oracle diagnostic after primary evaluation; not new model tests',a_grid=np.round(np.arange(.5,.981,.01),6).tolist(),observation_std_grid=np.round(np.arange(.05,.751,.025),6).tolist(),delays=[1,2,3],physical_source_stationary_variance=.25/(1-.8**2),other_parameters='b.7 c.5 alpha.5 beta.8 qZ=qX=.25; qY adjusted to hold source marginal variance constant',reported='sqrt(1-P_filter/P_prior), theoretical correlation of conditional mean with true Gaussian state',unchanged_primary_metrics=True);(ROOT/'ANALYTICAL_MAP_REGISTERED.json').write_text(json.dumps(dict(protocol=protocol,source_sha256=hashlib.sha256((ROOT/'analytical_map.py').read_bytes()).hexdigest()),indent=2));rows=[];var=.25/(1-.8**2)
 for d in [1,2,3]:
  for a in protocol['a_grid']:
   qy=var*(1-a*a)
   for obs in protocol['observation_std_grid']:
    theta=[a,.7,.5,.5*np.sqrt(qy/.25),.4,.25,obs*obs];A,Q,R,P=plant(theta,d);H=np.eye(len(A))[2:3];PP=solve_discrete_are(A.T,H.T,Q,np.array([[R]]));F=PP-np.outer(PP[:,2],PP[2])/(PP[2,2]+R);frac=np.diag(F)[:2]/np.diag(P)[:2]
    rows.append(dict(delay=d,a=a,observation_std=obs,physical_qY=qy,Y_corr=float(np.sqrt(max(0,1-frac[0]))),Z_corr=float(np.sqrt(max(0,1-frac[1]))),Y_remaining_variance=float(frac[0]),Z_remaining_variance=float(frac[1])))
  print('Mapped delay',d,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'P0_analytical_map.csv',index=False)
 # Independent Riccati iteration agreement at the preregistered baseline.
 p=plant([.8,.7,.5,.5,.4,.25,.0625],1);A,Q,R,P=p;H=np.eye(3)[2:3];PP=solve_discrete_are(A.T,H.T,Q,np.array([[R]]));F=PP-np.outer(PP[:,2],PP[2])/(PP[2,2]+R);v=infer(np.zeros((1,500)),p)['cov'][-1];err=float(np.max(abs(F-v)));assert err<1e-10;(OUT/'analytical_map_check.json').write_text(json.dumps(dict(points=len(rows),Riccati_vs_filter_max_error=err,status='PASS'),indent=2));print('Analytic map',len(rows),'points; numerical agreement',err)
if __name__=='__main__':main()
