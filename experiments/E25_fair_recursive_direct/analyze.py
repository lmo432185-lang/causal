"""Pre-specified paired analysis; never adjusts models or confirmation decisions."""
import json
import numpy as np,pandas as pd
from scipy.stats import t,ttest_1samp,norm
from settings import *
from confirm import holm
from model import load

def interval(a,level=.95):
 a=np.asarray(a);mu=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)));r=float(t.ppf((1+level)/2,len(a)-1)*se);return mu,mu-r,mu+r

def bootstrap(groups,seed):
 # Equal conditions, paired model scores, shared resampled training-seed blocks.
 # Within each sampled condition/seed, independent test trajectories are resampled.
 rng=np.random.default_rng(seed);groups=np.asarray(groups);g,s,n=groups.shape;result=[]
 for start in range(0,10000,250):
  ix=rng.integers(0,s,size=(250,s));v=np.zeros((250,s))
  for c in range(g):
   j=rng.integers(0,n,size=(250,s,n));v+=groups[c][ix[:,:,None],j].mean(-1)/g
  result.extend(v.mean(1))
 return np.quantile(result,[.025,.975]).tolist()

def wilson(k,n):
 z=norm.ppf(.975);p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return mid-r,mid+r

def main():
 summaries=[];paired=[];overall=[];efficiency=[];conditiontests=[];seedtables=[];timings=[]
 for batch in ['A','B']:
  df=pd.read_csv(OUT/f'test_{batch}_metrics.csv.gz');df['condition']=df.case.str.rsplit('_',n=1).str[0]
  summaries.append(df.groupby(['batch','condition','n','method']).mean(numeric_only=True).reset_index())
  key=['condition','seed','trajectory'];pivot=df.pivot(index=key+['n'],columns='method',values='NLL');pairs=[]
  for n in SIZES:
   at=pivot.xs(n,level='n');recur=at['recursive']
   for comp in ['direct_full','direct_fixed','direct_random','W1','full_W1','full_X']:
    delta=(at[comp]-recur).rename('gain').reset_index();delta['batch']=batch;delta['n']=n;delta['comparison']=comp;pairs.append(delta)
    for cond,group in delta.groupby('condition'):
     sm=group.groupby('seed').gain.mean();mu,lo,hi=interval(sm);l90,h90=interval(sm,.9)[1:];p=float(ttest_1samp(sm,.005,alternative='greater').pvalue)
     row=dict(batch=batch,n=n,condition=cond,comparison=comp,gain=mu,low95=lo,high95=hi,low90=l90,high90=h90,p_above_practical=p,sd_seed=float(sm.std(ddof=1)),seeds=len(sm));paired.append(row)
     if n==400 and comp.startswith('direct_'):conditiontests.append(row.copy())
    if comp.startswith('direct_'):
     groups=np.stack([sub.pivot(index='seed',columns='trajectory',values='gain').loc[SEEDS].to_numpy() for _,sub in delta.groupby('condition')]);sm=groups.mean((0,2));mu,lo,hi=interval(sm);l90,h90=interval(sm,.9)[1:];bl,bh=bootstrap(groups,250000+ord(batch)+n+len(comp))
     overall.append(dict(batch=batch,n=n,comparison=comp,gain=mu,seed_t_low95=lo,seed_t_high95=hi,bootstrap_low95=bl,bootstrap_high95=bh,low90=l90,high90=h90,p_above_practical=float(ttest_1samp(sm,.005,alternative='greater').pvalue),p_below_negative_practical=float(ttest_1samp(sm,-.005,alternative='less').pvalue),equivalent_90=(l90>-.005 and h90<.005),positive_dual_conditions=sum(z['gain']>0 for z in paired if z['batch']==batch and z['n']==n and z['comparison']==comp and z['condition'].startswith(('S2','S3','S4')))))
   # Data efficiency for each model vs same held-out B_full trained on400.
   ref=pivot.xs(400,level='n')['direct_full']
   for method in METHODS:
    diff=(ref-at[method]).rename('gain').reset_index()
    for cond,sub in diff.groupby('condition'):
     sm=sub.groupby('seed').gain.mean();mu,lo,hi=interval(sm,.9);efficiency.append(dict(batch=batch,n=n,method=method,condition=cond,gain_vs_direct400=mu,low90=lo,high90=hi,equivalent_90=lo>-.005 and hi<.005))
  seedtables.append(pd.concat(pairs).groupby(['batch','n','comparison','condition','seed']).gain.mean().reset_index())
 for job in json.loads((ROOT/'jobs.json').read_text()):
  for n in SIZES:
   m=load(OUT/job['case']/f'model_n{n}.json')
   for method,v in m['timing'].items():timings.append(dict(case=job['case'],seed=job['seed'],n=n,method=method,**v,process_peak_RSS_kib=m['peak_RSS_kib'],budget_pass=v['cpu']<=CPU_LIMIT))
 summary=pd.concat(summaries);summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'condition_comparisons.csv',index=False);o=pd.DataFrame(overall)
 for batch in ['A','B']:
  at=(o.batch==batch)&(o.n==400);o.loc[at,'p_holm3']=holm(o.loc[at,'p_above_practical'])
 o['practical_win']=(o.gain>=.005)&(o.bootstrap_low95>.005)&(o.p_holm3<.05)&(o.positive_dual_conditions>=6)
 o.to_csv(OUT/'overall_comparisons.csv',index=False);ct=pd.DataFrame(conditiontests)
 for batch in ['A','B']:
  at=ct.batch==batch;ct.loc[at,'p_holm33']=holm(ct.loc[at,'p_above_practical'])
 ct.to_csv(OUT/'condition_holm33.csv',index=False);pd.DataFrame(efficiency).to_csv(OUT/'data_efficiency.csv',index=False);pd.concat(seedtables).to_csv(OUT/'seed_comparisons.csv',index=False);timeframe=pd.DataFrame(timings);timeframe.to_csv(OUT/'timing.csv',index=False)
 cf=pd.read_csv(OUT/'confirmation.csv');cf['condition']=cf.case.str.rsplit('_',n=1).str[0];ex=cf.groupby(['condition','n','method','test']).accepted.agg(['sum','count']).reset_index();ex[['low95','high95']]=[wilson(int(r['sum']),int(r['count'])) for _,r in ex.iterrows()];ex.to_csv(OUT/'expansions.csv',index=False)
 primary=o[o.n==400];win=bool(primary.practical_win.all() and timeframe.budget_pass.all());full=primary[primary.comparison=='direct_full'];equiv=bool(full.equivalent_90.all());worse=bool((full.seed_t_high95<-.005).all());out=dict(primary_rows=primary.to_dict('records'),recursive_unique_value_demonstrated=win,global_equivalence_to_full_direct=equiv,global_practically_worse_than_full_direct=worse,CPU_budget_violations=int((~timeframe.budget_pass).sum()),scope='linear Gaussian, equal-condition average; no causal inference or E3',test_batches_independent_data_not_independent_retraining=True)
 (OUT/'CONCLUSION.json').write_text(json.dumps(out,indent=2));print(primary.to_string(index=False));print('unique',win,'equivalent',equiv,'worse',worse)
if __name__=='__main__':main()
