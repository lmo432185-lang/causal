from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CELLS
O=Path(__file__).parent/'results';d=pd.read_csv(O/'scan_metrics.csv.gz');s=d.groupby(['cell','state','method'])[['MSE','NMSE','corr','posterior_variance']].mean();s.to_csv(O/'scan_summary.csv');d.groupby(['cell','seed','state','method'])[['MSE','NMSE','corr','posterior_variance']].mean().to_csv(O/'scan_by_seed.csv');rng=np.random.default_rng(803);interval=[]
for (cell,state),g in d.groupby(['cell','state']):
 p=g.pivot(index=['seed','trajectory'],columns='method',values='MSE')
 for a,b in [('KF_known','L1_matched'),('RTS_future','KF_known'),('L1_matched','L1_frozen_OOD')]:
  v=(p[a]-p[b]).to_numpy().reshape(3,200);boot=np.array([np.take_along_axis(v,rng.integers(0,200,(3,200)),axis=1).mean() for _ in range(1500)]);interval.append(dict(cell=cell,state=state,method=a,reference=b,MSE_difference=float(v.mean()),CI95=np.quantile(boot,[.025,.975]).tolist()))
(O/'scan_paired_intervals.json').write_text(json.dumps(interval,indent=2));fig,axs=plt.subplots(2,5,figsize=(17,6),sharey='row');methods=['L1_frozen_OOD','L1_matched','KF_known','RTS_future'];factorlist=['source_gain','middle_gain','measurement_noise','source_speed','middle_memory'];base=dict(source_gain=.5,middle_gain=.8,measurement_noise=0.,source_speed=.8,middle_memory=.7)
for col,factor in enumerate(factorlist):
 cells=[('base',base[factor])]+[(c['name'],c['level']) for c in CELLS if c['factor']==factor];cells=sorted(cells,key=lambda v:v[1])
 for row,state in enumerate(['Y','Z']):
  ax=axs[row,col]
  for method in methods:ax.plot([v for _,v in cells],[s.loc[(n,state,method),'NMSE'] for n,_ in cells],marker='o',label=method)
  ax.set(title=f'{factor}: {state}',xlabel='Parameter',ylabel='Normalized reconstruction MSE' if col==0 else '')
axs[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(O/'conditions.png',dpi=160);plt.close(fig)
models=json.loads((O/'structure_models.json').read_text());pd.DataFrame(models).to_csv(O/'structure_model_summary.csv',index=False);st=pd.read_csv(O/'structure_state_metrics.csv.gz');st.groupby(['group','state'])[['MSE','NMSE','corr']].mean().to_csv(O/'structure_state_summary.csv');rank=pd.crosstab(pd.DataFrame(models).group,pd.DataFrame(models)['rank']);rank.to_csv(O/'predictive_rank_counts.csv');fig,ax=plt.subplots(figsize=(7,4));rank.plot.bar(ax=ax);ax.set(ylabel='Independent seed count',title='X-only selected predictive rank; not causal topology',xlabel='Generator label (revealed after fitting)');ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(O/'rank.png',dpi=160);plt.close(fig)
print(s.round(4).to_string());print(pd.DataFrame(models)[['group','seed','rank','test_future_MSE']].to_string(index=False));print(st.groupby(['group','state'])[['NMSE','corr']].mean().round(4).to_string())
