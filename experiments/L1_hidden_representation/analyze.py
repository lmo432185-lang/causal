from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent/'results';d=pd.read_csv(R/'trajectory_metrics.csv.gz');s=d.groupby(['task','method'])[['MSE','NMSE','corr']].mean();s.to_csv(R/'summary.csv');d.groupby(['seed','task','method'])[['MSE','NMSE','corr']].mean().to_csv(R/'summary_by_seed.csv')
rng=np.random.default_rng(311);rows=[]
for task in ['Y','Z','X_next']:
 g=d[d.task==task];p=g.pivot(index=['seed','trajectory'],columns='method',values='MSE')
 pairs=[('L_only','random_L'),('L_only','X_current'),('X_current_L','X_current'),('X_history_L','X_history')] if task!='X_next' else [('A2_history','A1_current'),('A3_history_L','A2_history'),('A_random_history','A2_history'),('A4_history_trueYZ','A2_history')]
 for a,b in pairs:
  v=(p[a]-p[b]).to_numpy().reshape(3,200);boot=np.array([np.take_along_axis(v,rng.integers(0,200,(3,200)),axis=1).mean() for _ in range(3000)]);rows.append(dict(task=task,method=a,reference=b,MSE_difference=float(v.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),relative_improvement=float(1-p[a].mean()/p[b].mean())))
(R/'paired_intervals.json').write_text(json.dumps(rows,indent=2));latent=[]
for seed in [101,202,303]:
 a=np.load(R/f'seed{seed}/test_predictions.npz');l=a['L'].reshape(-1,2);cov=np.cov(l.T);eig=np.linalg.eigvalsh(cov);latent.append(dict(seed=seed,L1_L2_correlation=float(np.corrcoef(l.T)[0,1]),first_PC_variance_fraction=float(eig[-1]/eig.sum())))
(R/'latent_redundancy.json').write_text(json.dumps(latent,indent=2))
fig,ax=plt.subplots(1,2,figsize=(10,4))
for j,k in enumerate(['Y','Z']):
 v=s.loc[k].loc[['random_L','X_current','L_only','X_current_L','X_history']];ax[j].bar(range(len(v)),v.NMSE);ax[j].set(xticks=range(len(v)),xticklabels=['Random L','Current X','Learned L','X + L','20-point X'],ylabel='MSE / validation truth variance',title=f'{k} reconstruction');ax[j].tick_params(axis='x',rotation=22)
fig.tight_layout();fig.savefig(R/'reconstruction.png',dpi=160);plt.close(fig)
a=np.load(R/'seed101/test_predictions.npz');fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
for j,k in enumerate(['Y','Z']):ax[j].plot(np.arange(19,499),a['truth'][0,:,j],label=f'True {k}',lw=1);ax[j].plot(np.arange(19,499),a['L_only'][0,:,j],label='Frozen L + validation linear probe',lw=1);ax[j].legend(fontsize=8);ax[j].set_ylabel(k)
ax[-1].set_xlabel('Time step; preselected seed 101, first test trajectory');fig.tight_layout();fig.savefig(R/'example.png',dpi=160);plt.close(fig)
print(json.dumps(latent,indent=2));print(json.dumps(rows,indent=2))
