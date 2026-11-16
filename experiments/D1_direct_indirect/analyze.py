from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent/'results'
def read(f):return pd.read_csv(R/f,keep_default_na=False,na_values=[''])
d=read('structure_state_metrics.csv.gz');d['forced_correct']=(d.truth==d.prediction).astype(int)
cols=['forced_correct','declared','correct_declared','wrong_declared','refused','wrong_middle','source_corr_filter','source_corr_smooth','source_NMSE_filter','source_NMSE_smooth','middle_corr_filter','middle_corr_smooth','L_Y_corr','L_Y_incremental_R2','L_W_corr','L_W_incremental_R2']
for keys,name in [(['part'],'part'),(['part','ni'],'noise'),(['part','cohort'],'cohort'),(['part','a','c','ni'],'conditions')]:d.groupby(keys)[cols].mean().to_csv(R/f'summary_{name}.csv')
pd.crosstab(d.truth,d.prediction,normalize='index').to_csv(R/'confusion.csv')
d[d.part=='A'].groupby(['beta','ni'])[['beta_direct_est','gamma_indirect_est','forced_correct','declared']].mean().to_csv(R/'beta_trend.csv')
d['gamma_truth']=d.alpha*d.beta
d[d.part=='B'].groupby(['gamma_truth','ni'])[['gamma_indirect_est','beta_direct_est','forced_correct','declared']].mean().to_csv(R/'gamma_trend.csv')
t=read('timing_metrics.csv.gz');tc=['valid','normal_false_alarm','early','delay','lead','cover_1','late_forecast_MSE','paired_null_MSE']
t.groupby(['mode','method'])[tc].mean().to_csv(R/'summary_timing.csv');t.groupby(['mode','method','noise'])[tc].mean().to_csv(R/'summary_timing_noise.csv')
# Event/trajectory, not method rows, are independent sampling units. Wilson intervals
# are descriptive conditional on frozen training/calibration and this balanced design.
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [float(mid-rad),float(mid+rad)]
ci=[]
for (mode,meth),g in t.groupby(['mode','method']):
 for m in (['normal_false_alarm'] if mode=='null' else ['valid','cover_1']):
  k=int(g[m].sum());ci.append(dict(mode=mode,method=meth,metric=m,k=k,n=len(g),rate=k/len(g),wilson95=wilson(k,len(g))))
(R/'timing_intervals.json').write_text(json.dumps(ci,indent=2))
# Stratified bootstrap across trajectories within cell for aggregate state correlations.
rng=np.random.default_rng(91);boot=[]
for part,g in d.groupby('part'):
 cells=[v for _,v in g.groupby('cell')]
 for metric in ['source_corr_filter','source_corr_smooth','L_Y_corr','L_Y_incremental_R2']:
  vals=np.stack([v[metric].to_numpy() for v in cells]);est=np.array([np.take_along_axis(vals,rng.integers(0,vals.shape[1],vals.shape),axis=1).mean() for _ in range(1000)]);boot.append(dict(part=part,metric=metric,mean=float(vals.mean()),CI95=np.quantile(est,[.025,.975]).tolist()))
(R/'state_intervals.json').write_text(json.dumps(boot,indent=2))
fig,ax=plt.subplots(1,2,figsize=(10,4));s=d.groupby('part')[['forced_correct','correct_declared','refused']].mean();s.plot.bar(ax=ax[0],rot=0);ax[0].set(ylabel='Fraction',ylim=(0,1.08),title='X-only structure, 80-unit records');ax[0].legend(fontsize=7);im=ax[1].imshow(pd.crosstab(d.truth,d.prediction,normalize='index'),vmin=0,vmax=1);ax[1].set(xticks=range(4),yticks=range(4),xticklabels=['none','direct','indirect','mixed'],yticklabels=['none','direct','indirect','mixed'],xlabel='Forced candidate',ylabel='True generator',title='Forced classification');fig.colorbar(im,ax=ax[1]);fig.tight_layout();fig.savefig(R/'structure.png',dpi=170);plt.close(fig)
e=np.load(R/'timing_examples.npz');time=np.arange(1281)*.025;fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
for ax,mode in zip(axs,['source_mean','beta_loss']):
 ax.plot(time,e[f'n1_{mode}_states'][:,0],label='True Y');ax.plot(time,e[f'n1_{mode}_Yfilter'],label='Causal Y estimate');ax.plot(time,e[f'n1_{mode}_Ysmooth'],label='Retrospective Y estimate',alpha=.7);ax.axvline(e[f'n1_{mode}_T'][0],color='k',ls='--');ax.set(title=f'Preselected example: {mode}, medium noise',ylabel='Source units');ax.legend(fontsize=8)
axs[-1].set_xlabel('Model time');fig.tight_layout();fig.savefig(R/'timing.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));d.groupby('part')[['source_corr_filter','source_corr_smooth','L_Y_corr']].mean().plot.bar(ax=ax,rot=0);ax.set(ylabel='Mean within-trajectory correlation',title='State information does not establish graph identity');fig.tight_layout();fig.savefig(R/'states.png',dpi=170);plt.close(fig)
print(d.groupby('part')[cols[:5]+['source_corr_filter','source_corr_smooth','L_Y_corr','L_Y_incremental_R2']].mean().to_string());print(t.groupby(['mode','method'])[tc].mean().to_string());print('Counts',d.groupby('part').size().to_dict(), 'equivalence',len(read('equivalence_metrics.csv.gz')), 'timing',len(t));print('declarations',d[d.declared==1].groupby(['part','truth','prediction']).size().to_dict())
