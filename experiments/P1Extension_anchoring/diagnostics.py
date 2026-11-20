"""Supplementary post-freeze physical-unit parameter diagnostics, not new model fitting."""
import json,time
import numpy as np,pandas as pd
from settings import ROOT,DATA,OUT

def main():
 p=pd.read_csv(OUT/'estimated_parameters.csv');truth={v['case']:v['params'] for v in json.loads((DATA/'sealed_manifest.json').read_text())};rows=[]
 for _,v in p[p.method=='E4YZ'].iterrows():
  target=truth[v['case']];converted=dict(a=v.a,b=v.b,c=v.c,u=v.u*v.hZ/v.hY,v=v.v/v.hZ)
  for k,estimate in converted.items():rows.append(dict(case=v['case'],tag=v.tag,policy=v.policy,seed=v.seed,parameter=k,true=target[k],estimated=estimate,absolute_error=abs(estimate-target[k])))
 pd.DataFrame(rows).to_csv(OUT/'E4YZ_physical_parameter_errors.csv',index=False);w=np.linspace(0,np.pi,4096);z=np.exp(-1j*w);a,b,c,u,v=.8,.7,.5,.5,.4;signals=[u*v*z**(1+dy+dx)/((1-a*z)*(1-b*z)*(1-c*z)) for dy,dx in [(1,3),(2,2),(3,1)]];err=max(float(np.max(abs(s-signals[0]))) for s in signals);assert err==0
 (OUT/'independent_delay_counterexample.json').write_text(json.dumps(dict(time=time.time(),scope='supplementary mathematical counterexample, not new empirical training',edge_delay_pairs=[[1,3],[2,2],[3,1]],known_input_transfer_max_difference=err,autonomous_X_spectrum='independent of both pure delays',interpretation='known input alone need not identify individual edge delays when common-delay assumption is removed'),indent=2))
if __name__=='__main__':main()
