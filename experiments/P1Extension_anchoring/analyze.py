import json,time
import numpy as np,pandas as pd
from settings import ROOT,OUT,METHODS

def main():
 c=pd.read_csv(OUT/'case_metrics.csv');t=pd.read_csv(OUT/'trajectory_metrics.csv.gz');ident=pd.read_csv(OUT/'identification.csv');keys=['tag','policy','method','state'];num=c.select_dtypes('number').columns.difference(['seed']);c.groupby(keys)[num].agg(['mean','min','max']).to_csv(OUT/'summary.csv');gates=[]
 for (case,m),q in c[c.method.isin(METHODS)].groupby(['case','method']):
  z=q.set_index('state');ref='C_XI' if m=='E3' else 'C_X';oracle=c[(c.case==case)&(c.method==ref)].set_index('state');di=ident[(ident.case==case)&(ident.method==m)].iloc[0];checks={}
  for st in ['Y','Z']:
   checks[st+'_coverage']=bool(abs(z.loc[st,'cover90']-.9)<=.03);checks[st+'_PIT']=bool(z.loc[st,'PIT_KS']<=.05);checks[st+'_MSE']=bool(z.loc[st,'MSE']/oracle.loc[st,'MSE']<=1.2)
  checks['joint_coverage']=bool(abs(z.loc['joint','cover90']-.9)<=.03);checks['joint_NLL']=bool(z.loc['joint','NLL']-oracle.loc['joint','NLL']<=.2);conditional=all(checks.values());orderok=bool(di.valid_order) if m=='E1' else True;identify=bool(di.known_family_unique and di.unique_delay_correct and di.validation_unique_delay_correct and orderok);gates.append(dict(case=case,method=m,tag=q.tag.iloc[0],policy=q.policy.iloc[0],seed=int(q.seed.iloc[0]),**checks,conditional_probability_pass=conditional,known_family_identification_pass=identify,physical_scale_both_anchored=m=='E4YZ',candidate_conditional_P2_pass=bool(conditional and identify and m=='E4YZ'),not_global_proof=True))
 gates=pd.DataFrame(gates);gates.to_csv(OUT/'P2_gate_cases.csv',index=False);required=gates[gates.tag.isin(['baseline','rich'])&(gates.policy=='passive')];decisions=[]
 for m in METHODS:
  a=required[required.method==m];decisions.append(dict(method=m,required_cases=len(a),probability_passes=int(a.conditional_probability_pass.sum()),identification_passes=int(a.known_family_identification_pass.sum()),combined_passes=int(a.candidate_conditional_P2_pass.sum()),meets_primary_candidate_gate=bool(a.candidate_conditional_P2_pass.all()),conditional_only=True))
 (OUT/'P2_DECISION.json').write_text(json.dumps(dict(time=time.time(),scope='candidate-specific recommendation only, no P2 experiment run',groups=decisions,limitations=['known positive chain assumed, not discovered','same delay on both links assumed','finite equivalence-family and local-rank checks are not unrestricted global proof','E4 uses sparse train/validation hidden measurements; not pure unsupervised X-only']),indent=2));ident.groupby(['tag','policy','method'])[['forced_delay_correct','unique_delay_correct','validation_unique_delay_correct','retained_known_equivalents','known_family_unique']].mean().to_csv(OUT/'identification_summary.csv')
 # Hierarchical paired seed/trajectory bootstrap for principal contrasts versus E0.
 rng=np.random.default_rng(20511202);res=[]
 for tag in ['baseline','rich','delay2','delay3','weak','wrong_order_control']:
  for policy in ['passive','active']:
   for m in ['E1','E2','E3','E4Y','E4Z','E4YZ']:
    for state,metric,op in [('Y','MSE','ratio'),('Z','MSE','ratio'),('joint','NLL','difference')]:
     a=t[(t.tag==tag)&(t.policy==policy)&(t.method==m)&(t.state==state)].sort_values(['seed','trajectory']);b=t[(t.tag==tag)&(t.policy==policy)&(t.method=='E0')&(t.state==state)].sort_values(['seed','trajectory']);assert len(a)==600 and len(b)==600;aa=a[metric].to_numpy().reshape(3,200);bb=b[metric].to_numpy().reshape(3,200);si=rng.integers(0,3,(2000,3));ti=rng.integers(0,200,(2000,3,200));av=aa[si[:,:,None],ti].mean((1,2));bv=bb[si[:,:,None],ti].mean((1,2));v=av/bv if op=='ratio' else av-bv;lo,hi=np.quantile(v,[.025,.975]);point=aa.mean()/bb.mean() if op=='ratio' else aa.mean()-bb.mean();res.append(dict(tag=tag,policy=policy,method=m,state=state,metric=metric,comparison=op,estimate=float(point),low95=float(lo),high95=float(hi),independent_data_seeds=3,test_trajectories=600))
 pd.DataFrame(res).to_csv(OUT/'paired_intervals.csv',index=False)
 print('Analysis complete',len(c),'case metrics;',len(gates),'gates;',len(res),'paired intervals')
if __name__=='__main__':main()
