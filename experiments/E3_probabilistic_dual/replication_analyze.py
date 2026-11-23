import json,numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from settings import ROOT,OUT
from analyze import CONTRASTS,interval,bootstrap
from confirm import holm

def analyze_frame(df,label):
 df=df.copy();df['condition']=df.case.str.rsplit('_',n=1).str[0];p=df.pivot(index=['condition','seed','trajectory'],columns='group',values='NLL');rows=[]
 for index,(contrast,co) in enumerate(CONTRASTS.items()):
  delta=sum(p[g]*v for g,v in co.items()).rename('gain').reset_index();a=np.stack([z.pivot(index='seed',columns='trajectory',values='gain').to_numpy() for c,z in delta.groupby('condition')]);seed=a.mean((0,2));mu,lo,hi=interval(seed);bl,bh=bootstrap(a,880100+index+len(label));rows.append(dict(analysis=label,contrast=contrast,gain=mu,low95=lo,high95=hi,bootstrap_low95=bl,bootstrap_high95=bh,p=float(ttest_1samp(seed,.005,alternative='greater').pvalue),independent_training_seeds=a.shape[1]))
 q=pd.DataFrame(rows);q['p_holm8']=holm(q.p);q['practical_success']=(q.bootstrap_low95>.005)&(q.p_holm8<.05);return q

def main():
 allrows=[];summary=[];convergence=[]
 for batch in ['A','B']:
  frames={}
  for budget in ['em5','em25']:
   root=ROOT/'replication'/budget;assert (root/f'TEST_{batch}_FROZEN.json').exists();d=pd.read_csv(root/f'test_{batch}.csv.gz');frames[budget]=d;allrows.append(analyze_frame(d,f'new10_{budget}_{batch}'));summary.append(d.groupby('group').mean(numeric_only=True).reset_index().assign(budget=budget,batch=batch,scope='new10'))
  old=pd.read_csv(OUT/f'test_{batch}.csv.gz');old=old[old.n==100];allrows.append(analyze_frame(pd.concat([old,frames['em5']]),f'combined30_em5_{batch}'))
  p5=frames['em5'].pivot(index=['case','seed','trajectory'],columns='group',values='NLL');p25=frames['em25'].pivot(index=['case','seed','trajectory'],columns='group',values='NLL')
  for group in ['B','D','G']:
   delta=(p5[group]-p25[group]).groupby('seed').mean();mu,lo,hi=interval(delta);convergence.append(dict(batch=batch,comparison='EM5_minus_EM25_'+group,gain=mu,low95=lo,high95=hi))
 rows=pd.concat(allrows);rows.to_csv(OUT/'replication_contrasts.csv',index=False);pd.concat(summary).to_csv(OUT/'replication_summary.csv',index=False);pd.DataFrame(convergence).to_csv(OUT/'convergence_diagnostic.csv',index=False);r=rows[rows.contrast.isin(['recursive_update','interaction','vs_strong_direct'])];(OUT/'REPLICATION_CONCLUSION.json').write_text(json.dumps(dict(new10_rows=r.to_dict('records'),primary20_results_preserved=True,combined30_is_supplementary=True),indent=2));print(r.to_string(index=False));print(pd.DataFrame(convergence).to_string(index=False))
if __name__=='__main__':main()
