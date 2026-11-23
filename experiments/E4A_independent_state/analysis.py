"""Pre-registered aggregate comparisons. Never tune or refit models."""
import json,time,hashlib,numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from settings import *
from confirm import holm
CONTRASTS={'recursive_gain':{'M3':1,'M4':-1},'direct_update':{'M1':1,'M3':-1},'recursive_update':{'M2':1,'M4':-1},'interaction':{'M2':1,'M4':-1,'M1':-1,'M3':1}}
def interval(z,seed=771,n=10000):
 # Independent training seed is the resampling block. Keep all conditions paired.
 rng=np.random.default_rng(seed);blocks=[g.delta.to_numpy() for _,g in z.groupby('seed',sort=True)];means=np.array([b.mean() for b in blocks]);chosen=rng.integers(0,len(blocks),(n,len(blocks)));resampled=np.empty_like(chosen,dtype=float)
 for i,block in enumerate(blocks):
  mask=chosen==i;count=int(mask.sum());resampled[mask]=rng.choice(block,size=(count,len(block)),replace=True).mean(1)
 draw=resampled.mean(1)
 return float(np.mean(means)),*np.quantile(draw,[.025,.975]).tolist(),float(ttest_1samp(means,.005,alternative='greater').pvalue)

def main():
 for b in ['A','B']:
  m=json.loads((ROOT/f'TEST_{b}_FROZEN.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in m['files'].items())
 tables=[];rows=[]
 for batch in ['A','B']:
  d=pd.read_csv(OUT/f'test_{batch}.csv.gz');d['condition']=d.case.str.split('_').str[0];tables.append(d)
  p=d.pivot(index=['condition','case','seed','length','trajectory'],columns='group',values='NLL').reset_index()
  for name,coef in {**CONTRASTS,'history_gain':{'Xhistory':1,'M4':-1}}.items():
   p['delta']=sum(p[g]*c for g,c in coef.items())
   for length in LENGTHS:
    z=p[p.length==length]
    for cond in ['ALL']+sorted(z.condition.unique()):
     v=z if cond=='ALL' else z[z.condition==cond];mean,lo,hi,pv=interval(v,seed=771+length)
     rows.append(dict(batch=batch,length=length,condition=cond,contrast=name,gain=mean,lo=lo,hi=hi,p=pv,seeds=v.seed.nunique(),trajectories=len(v)))
 r=pd.DataFrame(rows);r['p_holm']=np.nan
 for (batch,length),g in r.groupby(['batch','length']):
  for overall in [True,False]:
   ix=g.index[(g.condition=='ALL')==overall];ix=ix[r.loc[ix,'contrast']!='history_gain'];r.loc[ix,'p_holm']=holm(r.loc[ix,'p'])
 r.to_csv(OUT/'contrasts.csv',index=False);d=pd.concat(tables);d.groupby(['batch','length','condition','group']).agg(NLL=('NLL','mean'),CRPS=('CRPS','mean'),cover90=('cover90','mean'),width90=('width90','mean'),MSE=('MSE','mean'),working_factor_count=('working_factor_count','mean'),Markov_dimension=('Markov_dimension','mean'),two_mode_alias_mass=('two_mode_alias_mass','mean')).reset_index().to_csv(OUT/'group_summary.csv',index=False)
 c=pd.read_csv(OUT/'confirmation_checks.csv');c['condition']=c.case.str.split('_').str[0];summ=[]
 for (length,cond),g in c.groupby(['length','condition']):
  k=int(g.novel_predictive_support.sum());n=len(g);p=k/n;den=1+1.96**2/n;mid=(p+1.96**2/(2*n))/den;rad=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/den;summ.append(dict(length=length,condition=cond,support=k,n=n,rate=p,wilson_lo=mid-rad,wilson_hi=mid+rad,valid_proposals=int(g.proposal_valid.sum()),matched=int(g.pole_match.sum())))
 pd.DataFrame(summ).to_csv(OUT/'confirmation_summary.csv',index=False)
 (ROOT/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),no_refitting=True,bootstrap=10000,primary_length=5000),indent=2))
if __name__=='__main__':main()
