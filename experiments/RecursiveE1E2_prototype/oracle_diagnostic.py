"""Post-unblinding analytic predictive-rank reference; never used for model selection."""
import json
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from settings import ROOT,DATA,OUT,FULL,H
from core import whitening

def main():
 items=json.loads((DATA/'sealed_manifest.json').read_text())+json.loads((DATA/'sealed_positive_manifest.json').read_text());rows=[]
 for meta in items:
  if meta['seed']!=101:continue
  A=np.array(meta['A']);Q=np.diag(meta['Q']) if 'Q' in meta else np.eye(len(A))*.25;P=solve_discrete_lyapunov(A,Q);g=np.array([(np.linalg.matrix_power(A,k)@P)[-1,-1] for k in range(FULL+H+1)]);past=np.arange(1,FULL+1);future=np.arange(1,H+1);Cp=g[abs(past[:,None]-past[None,:])]-np.outer(g[past],g[past])/g[0];Cf=g[abs(future[:,None]-future[None,:])]-np.outer(g[future],g[future])/g[0];cross=g[past[:,None]+future[None,:]]-np.outer(g[past],g[future])/g[0];s=np.linalg.svd(whitening(Cp)@cross@whitening(Cf),compute_uv=False);info=-.5*np.log1p(-np.clip(s,0,1-1e-12)**2)/H
  for k,(rho,value) in enumerate(zip(s,info),1):rows.append(dict(system_index=meta['index'],system=meta['name'],predictive_direction=k,canonical_correlation=float(rho),ideal_information_gain_nats_per_step=float(value),scope='exact finite40-history Gaussian predictive reference after revealing generator; not trained model performance'))
 pd.DataFrame(rows).to_csv(OUT/'oracle_predictive_directions.csv',index=False);print(pd.DataFrame(rows).query('predictive_direction<=3').to_string(index=False))
if __name__=='__main__':main()
