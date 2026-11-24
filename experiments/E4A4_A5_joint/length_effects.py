"""Descriptive paired training-prefix effects on the same held-out X; no model choice."""
import numpy as np,pandas as pd
from settings import *
def main():
 assert (ROOT/'short/ANALYSIS_A_COMPLETE.json').exists()
 rows=[]
 for batch in ['A']:
  long=pd.read_csv(OUT/f'test_{batch}.csv.gz');short=pd.read_csv(ROOT/'short/results'/f'test_{batch}.csv.gz');z=long.merge(short,on=['case','seed','trajectory','method'],suffixes=('_5000','_1000'),validate='one_to_one');z['condition']=z.case.str.split('_').str[0];z['delta']=z.NLL_1000-z.NLL_5000
  for method,g in z.groupby('method'):
   for condition in ['ALL']+sorted(g.condition.unique()):
    a=g if condition=='ALL' else g[g.condition==condition];b=a.groupby('seed').delta.mean().to_numpy();rng=np.random.default_rng(314981);v=rng.choice(b,size=(10000,len(b)),replace=True).mean(1);lo,hi=np.quantile(v,[.025,.975]);rows.append(dict(batch=batch,condition=condition,method=method,gain_5000_over_1000=float(b.mean()),lo=float(lo),hi=float(hi),paired_trajectories=len(a),independent_training_seeds=len(b),hypothesis_test='descriptive_only_no_selection'))
 pd.DataFrame(rows).to_csv(OUT/'paired_length_effects.csv',index=False)
if __name__=='__main__':main()
