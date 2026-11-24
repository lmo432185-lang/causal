"""Descriptive diagnostics, not a replacement hypothesis family."""
import json,gzip,numpy as np,pandas as pd
from settings import *
def main():
 rows=[];runtime=[];fails=[];paths=[];mc=[]
 for j in json.loads((ROOT/'jobs.json').read_text()):
  with gzip.open(OUT/(j['case']+'.json.gz'),'rt') as f:b=json.load(f)
  runtime.append(dict(case=j['case'],cpu=b['cpu'],peak_RSS_kib=b['peak_RSS_kib']))
  for bank,diag in b['diagnostics'].items():
   paths.append(dict(case=j['case'],bank=bank,path_valid=diag['path_valid'],residual_r=diag['residual_r']))
   for route,info in diag['cpu'].items():
    m=b['models'].get(bank+'_'+route)
    rows.append(dict(case=j['case'],bank=bank,route=route,available=m is not None,converged=bool(m.get('converged',False)) if m else False,iterations=m.get('iterations',0) if m else 0,cpu=info['cpu'],failed_starts=len(info['failures']),stability_projections=m.get('stability_projections',0) if m else 0))
   for route,reason in diag.get('failures',{}).items():fails.append(dict(case=j['case'],bank=bank,route=route,reason=str(reason)))
  p=OUT/'bayes'/(j['case']+'.json.gz')
  if p.exists():
   with gzip.open(p,'rt') as f:z=json.load(f)
   mc.append(dict(case=j['case'],converged=z['converged'],Rhat_max=max(z['rhat']),ESS_min=min(z['ESS']),acceptance_min=min(z['acceptance']),acceptance_max=max(z['acceptance']),optimizer_success=z['optimizer_success'],cpu=z['cpu']))
 for data,name in [(rows,'optimization_summary'),(runtime,'runtime_summary'),(fails,'fit_failures'),(paths,'proposal_summary'),(mc,'mcmc_summary')]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
 print('Descriptive diagnostics saved')
if __name__=='__main__':main()
