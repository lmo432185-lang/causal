from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent/'results';d=pd.read_csv(O/'trajectory_metrics.csv.gz');cols=['declared','correct_declared','wrong_declared','refused','forced_correct','posterior'];g=d.groupby(['pool','info','length','prior_valid']);g[cols].mean().to_csv(O/'summary.csv');d.groupby(['pool','info','length','noise','prior_valid'])[cols].mean().to_csv(O/'summary_noise.csv');d.groupby(['pool','info','length','seed','prior_valid'])[cols].mean().to_csv(O/'summary_seed.csv')
# Wilson descriptive intervals; models and parameters fixed. Prefixes/conditions are paired.
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;m=(p+z*z/(2*n))/den;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [float(m-r),float(m+r)]
rows=[]
for (pool,info,length,noise,valid),v in d.groupby(['pool','info','length','noise','prior_valid']):
 for key in ['correct_declared','wrong_declared','refused']:
  n=len(v);k=int(v[key].sum());rows.append(dict(pool=int(pool),info=info,length=int(length),noise=float(noise),prior_valid=bool(valid),metric=key,n=n,k=k,rate=k/n,wilson95=wilson(k,n)))
(O/'rate_intervals.json').write_text(json.dumps(rows,indent=2));proof=json.loads((O/'pretest_law_comparisons.json').read_text());p=pd.DataFrame(proof['rows']);p.to_csv(O/'law_comparison_table.csv',index=False);fig,ax=plt.subplots(1,2,figsize=(12,4.8))
infos=['C0_X','C1_XZ','C2_Y','C2_Z','C2_X_negative'];s=d[(d.length==500)&(d.pool.isin([1,2,3]))].groupby(['pool','info']).correct_declared.mean().unstack('info').reindex(columns=infos);s.index=['Direct vs indirect','Mixed cancellation','Clock roles'];s.plot.bar(ax=ax[0],rot=15);ax[0].set(ylabel='Correct declaration / all tests',ylim=(0,1.08),title='Known-candidate reference, 500 observations');ax[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3)
law=p[p['info'].isin(infos)].pivot(index='pair',columns='info',values='observationally_equivalent').reindex(columns=infos);ax[1].imshow(law.astype(int),vmin=0,vmax=1,cmap='RdYlGn_r');ax[1].set(xticks=range(5),xticklabels=['X','X+Z','uY','uZ','uX'],yticks=range(3),yticklabels=['Direct vs indirect','Mixed cancellation','Clock roles'],title='1: same output law remains; 0: different')
for i in range(3):
 for j in range(5):ax[1].text(j,i,str(int(law.iloc[i,j])),ha='center',va='center')
fig.tight_layout();fig.savefig(O/'information_effect.png',dpi=170);plt.close(fig)
# Theoretical deterministic response examples, no test case selection.
from physics import controlled_mean
from config import PAIRS
fig,axs=plt.subplots(3,3,figsize=(12,8),sharex=True)
for i,pair in enumerate(PAIRS):
 for j,info in enumerate(['C2_Y','C2_Z','C2_X_negative']):
  for name,par in pair['models']:axs[i,j].plot(controlled_mean(par,info)[:,0],label=name)
  axs[i,j].set(title=f"{pair['name']}\n{info}",ylabel='Expected X')
  if i==2:axs[i,j].set_xlabel('Time')
  axs[i,j].legend(fontsize=6)
fig.tight_layout();fig.savefig(O/'intervention_means.png',dpi=150);plt.close(fig)
print(d.groupby(['pool','info','length','prior_valid'])[cols].mean().round(4).to_string());print('law table');print(p[['pair','info','observationally_equivalent','controlled_mean_max_difference']].to_string(index=False))
# Paired gains, stratified by measurement noise: repeated noise versions are never
# pooled as independent trajectories in confidence intervals.
rng=np.random.default_rng(950001);paired=[]
for pool in [0,1,2,3]:
 for sigma in [.05,.25,.75]:
  g=d[(d.pool==pool)&(d.noise==sigma)&(d.length==500)];v=g.pivot(index=['seed','truth_idx','trajectory'],columns='info',values='correct_declared');strata=len(v)//200
  for info in ['C1_XZ','C2_Y','C2_Z']:
   diff=(v[info]-v['C0_X']).to_numpy().reshape(strata,200);bs=np.array([np.take_along_axis(diff,rng.integers(0,200,diff.shape),axis=1).mean() for _ in range(1000)]);paired.append(dict(pool=pool,noise=sigma,info=info,minus='C0_X',coverage_difference=float(diff.mean()),CI95=np.quantile(bs,[.025,.975]).tolist()))
(O/'paired_coverage_intervals.json').write_text(json.dumps(paired,indent=2))
