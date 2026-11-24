import json,time,numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from settings import *
from confirm import holm
MAIN={'residual_gain':{'R0':1,'R2':-1},'joint_gain':{'R0':1,'R3':-1},'shared_vs_old_mix':{'J0':1,'J1':-1},'posterior_vs_point':{'J1':1,'J2':-1},'feedback_gain':{'C':1,'D':-1},'complement_gain':{'BestSingle':1,'E':-1}}
SECOND={'dense_parameter_gain':{'J1':1,'J1_param':-1},'structure_gain':{'J1_param':1,'J2':-1},'parameter_with_structures':{'J2_point':1,'J2':-1},'full_direct_reference':{'StrongestSingle':1,'E':-1}}
def interval(g):
 a=g.groupby('seed').delta.mean().to_numpy();r=np.random.default_rng(6511);v=r.choice(a,size=(10000,len(a)),replace=True).mean(1);return a.mean(),*np.quantile(v,[.025,.975]),float(ttest_1samp(a,.005,alternative='greater').pvalue)
def main():
 rows=[];summ=[]
 for batch in ['A','B']:
  d=pd.read_csv(OUT/f'test_{batch}.csv.gz');d['condition']=d.case.str.split('_').str[0];p=d.pivot(index=['condition','case','seed','trajectory'],columns='method',values='NLL').reset_index()
  for name,coef in {**MAIN,**SECOND}.items():
   p['delta']=sum(p[k]*v for k,v in coef.items())
   for cond in ['ALL']+sorted(p.condition.unique()):
    g=p if cond=='ALL' else p[p.condition==cond];valid=g[np.isfinite(g.delta)];mean,lo,hi,pv=interval(valid);rows.append(dict(batch=batch,condition=cond,contrast=name,family='main' if name in MAIN else 'secondary',gain=mean,lo=lo,hi=hi,p=pv,total_pairs=len(g),valid_pairs=len(valid),failed_pairs=len(g)-len(valid)))
  q=d.groupby(['condition','method']).mean(numeric_only=True).reset_index();q['batch']=batch;summ.append(q)
 r=pd.DataFrame(rows);r['p_holm']=np.nan
 for _,g in r.groupby(['batch','family']):
  for total in [True,False]:ix=g.index[(g.condition=='ALL')==total];r.loc[ix,'p_holm']=holm(r.loc[ix,'p'])
 r.to_csv(OUT/'contrasts.csv',index=False);pd.concat(summ).to_csv(OUT/'prediction_summary.csv',index=False);s=pd.read_csv(OUT/'confirmation_support.csv');s['condition']=s.case.str.split('_').str[0];s.groupby(['condition','route']).agg(support=('supported','sum'),n=('supported','size')).reset_index().to_csv(OUT/'support_summary.csv',index=False);(ROOT/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),secondary_training_length=1000,test_length=5000,not_independent_replication=True),indent=2))
if __name__=='__main__':main()
