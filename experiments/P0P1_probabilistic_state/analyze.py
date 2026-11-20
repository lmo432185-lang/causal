from pathlib import Path
import json
import numpy as np,pandas as pd
from settings import *

def main():
 c=pd.read_csv(OUT/'case_metrics.csv');t=pd.read_csv(OUT/'trajectory_metrics.csv.gz');numeric=c.select_dtypes('number').columns.difference(['seed','value']);c.groupby(['tag','factor','value','method','state'])[numeric].agg(['mean','min','max']).to_csv(OUT/'summary.csv');opt=[]
 for job in json.loads((ROOT/'public_jobs.json').read_text()):
  p=json.loads((OUT/job['case']/'parameters.json').read_text())
  for f in p['fits']:opt.append(dict(case=job['case'],start=f['start'],success=f['success'],nit=f['nit'],objective=f['objective'],validation_LL=f['validation_LL'],raw_gradient_max=f['raw_gradient_max'],**{k:v for k,v in zip(['a','b','c','alpha','beta_gauge','qx','R'],f['theta'])}))
 pd.DataFrame(opt).to_csv(OUT/'optimizer_summary.csv',index=False)
 # Entry criteria fixed before any data generation; labels used only to evaluate this gate.
 gate=[]
 for (case,method),g in c[c.method.isin(['B_d1','B_boot'])].groupby(['case','method']):
  z=g.set_index('state');r=c[(c.case==case)&(c.method=='C_filter')].set_index('state');checks={}
  for s in ['Y','Z']:
   checks[s+'_coverage']=bool(abs(z.loc[s,'cover90']-.9)<=.03);checks[s+'_PIT']=bool(z.loc[s,'PIT_KS']<=.05);checks[s+'_MSE']=bool(z.loc[s,'MSE']/r.loc[s,'MSE']<=1.2)
  checks['joint_coverage']=bool(abs(z.loc['joint','cover90']-.9)<=.03);checks['joint_NLL']=bool(z.loc['joint','NLL']-r.loc['joint','NLL']<=.2);gate.append(dict(case=case,method=method,tag=g.tag.iloc[0],seed=int(g.seed.iloc[0]),**checks,conditional_pass=all(checks.values()),physical_identity_identified=False,delay_identified=False,physical_P2_pass=False))
 gates=pd.DataFrame(gate);gates.to_csv(OUT/'P2_entry_gate.csv',index=False);primary=gates[(gates.method=='B_d1')&gates.tag.isin(['baseline','information_rich'])];(OUT/'P2_DECISION.json').write_text(json.dumps(dict(conditional_required_cases=6,conditional_passed=int(primary.conditional_pass.sum()),conditional_gate=bool(primary.conditional_pass.all()),physical_identity_gate=False,delay_gate=False,decision='DO NOT enter physical Y-Z chain P2; first establish anchoring/identifiability or explicitly change target to predictive function',bootstrap_primary_passed=int(gates[(gates.method=='B_boot')&gates.tag.isin(['baseline','information_rich'])].conditional_pass.sum()),no_new_thresholds=True),indent=2))
 # Hierarchical paired bootstrap: sample independent-fit seeds, then paired trajectories; never resample individual timepoints.
 rng=np.random.default_rng(2051120);intervals=[]
 for tag in sorted(c.tag.unique()):
  for method in ['B_d1','B_boot']:
   for state,metric,op in [('Y','MSE','ratio'),('Z','MSE','ratio'),('joint','NLL','difference')]:
    a=t[(t.tag==tag)&(t.method==method)&(t.state==state)].sort_values(['seed','trajectory']);b=t[(t.tag==tag)&(t.method=='C_filter')&(t.state==state)].sort_values(['seed','trajectory'])
    if len(a)!=600:continue
    assert a[['seed','trajectory']].to_numpy().tolist()==b[['seed','trajectory']].to_numpy().tolist();aa=a[metric].to_numpy().reshape(3,200);bb=b[metric].to_numpy().reshape(3,200);si=rng.integers(0,3,(2000,3));ti=rng.integers(0,200,(2000,3,200));av=aa[si[:,:,None],ti].mean((1,2));bv=bb[si[:,:,None],ti].mean((1,2));val=av/bv if op=='ratio' else av-bv;point=float(aa.mean()/bb.mean() if op=='ratio' else aa.mean()-bb.mean());lo,hi=np.quantile(val,[.025,.975]);intervals.append(dict(tag=tag,method=method,state=state,metric=metric,comparison=op,estimate=point,low95=float(lo),high95=float(hi),independent_fits=3,test_trajectories=600,bootstrap_replicates=2000))
 pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
 print('Analysis complete',len(c),'case-metric records;',len(gate),'gate records;',len(intervals),'paired intervals')
if __name__=='__main__':main()
