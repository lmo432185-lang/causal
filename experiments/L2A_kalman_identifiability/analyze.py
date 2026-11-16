from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from kalman import A,Q
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent/'results';d=pd.read_csv(R/'trajectory_metrics.csv.gz');g=d.groupby(['task','method']);s=g[['MSE','NMSE','corr','posterior_variance','posterior_95_coverage']].mean();s['RMSE']=np.sqrt(s.MSE);s.to_csv(R/'summary.csv');bs=d.groupby(['seed','task','method'])[['MSE','NMSE','corr','posterior_variance','posterior_95_coverage']].mean();bs['RMSE']=np.sqrt(bs.MSE);bs.to_csv(R/'summary_by_seed.csv');rng=np.random.default_rng(401);rows=[]
for task in ['Y','Z']:
 p=d[d.task==task].pivot(index=['seed','trajectory'],columns='method',values='MSE')
 for a,b in [('A4_Kalman_filter','A3_original_L'),('A5_RTS_smoother','A4_Kalman_filter'),('A4_Kalman_filter','A2_history20'),('D_known_model_history20','A4_Kalman_filter')]:
  v=(p[a]-p[b]).to_numpy().reshape(3,200);boot=np.array([np.take_along_axis(v,rng.integers(0,200,(3,200)),axis=1).mean() for _ in range(3000)]);rows.append(dict(task=task,method=a,reference=b,MSE_difference=float(v.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),relative_MSE_reduction=float(1-p[a].mean()/p[b].mean())))
(R/'paired_intervals.json').write_text(json.dumps(rows,indent=2));cov=np.load(R/'model_covariances.npz');P=solve_discrete_lyapunov(A,Q);theory=[]
for j,key in enumerate(['Y','Z']):
 pv=float(cov['filter'][19:499,j,j].mean());sv=float(cov['smoother'][19:499,j,j].mean());floor=.41 if key=='Y' else .3125;theory.append(dict(task=key,unconditional_variance=float(P[j,j]),expected_filter_MSE=pv,expected_smooth_MSE=sv,expected_filter_NMSE=float(pv/P[j,j]),expected_smooth_NMSE=float(sv/P[j,j]),population_filter_correlation=float(np.sqrt(1-pv/P[j,j])),population_smoother_correlation=float(np.sqrt(1-sv/P[j,j])),unseen_fresh_innovation_variance=floor,unseen_fresh_fraction_of_filter_uncertainty=floor/pv,remaining_filter_variance=pv-floor))
(R/'theoretical_reference.json').write_text(json.dumps(theory,indent=2));fig,ax=plt.subplots(1,2,figsize=(11,4))
for j,k in enumerate(['Y','Z']):
 v=s.loc[k].loc[['A1_current','A2_history20','A3_original_L','A4_Kalman_filter','A5_RTS_smoother']];ax[j].bar(range(5),v.MSE);ax[j].set(xticks=range(5),xticklabels=['Current X','20-point X','Original L','Kalman filter','RTS future'],ylabel='Test reconstruction MSE',title=k);ax[j].tick_params(axis='x',rotation=25)
fig.tight_layout();fig.savefig(R/'reconstruction.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
for j,k in enumerate(['Y','Z']):
 ax[j].plot(cov['filter'][:,j,j],label='Filter: X through current time');ax[j].plot(cov['smoother'][:,j,j],label='Smoother: all 500 X');ax[j].axhline(theory[j]['unseen_fresh_innovation_variance'],color='grey',ls=':',label='Fresh innovation causal floor');ax[j].set(ylabel=f'{k} posterior variance');ax[j].legend(fontsize=8)
ax[-1].set_xlabel('Retained trajectory time');fig.tight_layout();fig.savefig(R/'posterior_variance.png',dpi=170);plt.close(fig)
a=np.load(R/'seed101/estimates.npz');t=np.arange(19,499);fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
for j,k in enumerate(['Y','Z']):
 ax[j].plot(t,a['truth'][0,:,j],label='True state',lw=1,color='black',alpha=.7);ax[j].plot(t,a['A3_original_L'][0,:,j],label='Original L probe',lw=1);ax[j].plot(t,a['A4_Kalman_filter'][0,:,j],label='Kalman filter',lw=1,ls='--');ax[j].plot(t,a['A5_RTS_smoother'][0,:,j],label='RTS (future used)',lw=1);ax[j].set_ylabel(k);ax[j].legend(fontsize=8,ncol=4)
ax[-1].set_xlabel('Time: preselected seed 101, first test trajectory');fig.tight_layout();fig.savefig(R/'example.png',dpi=170);plt.close(fig)
print(s.round(6).to_string());print(json.dumps(theory,indent=2));print(json.dumps(rows,indent=2))
latent=[]
for seed in [101,202,303]:
 l=np.load(R/f'seed{seed}/estimates.npz')['L'].reshape(-1,2);eig=np.linalg.eigvalsh(np.cov(l.T));latent.append(dict(seed=seed,L1_L2_corr=float(np.corrcoef(l.T)[0,1]),PC1_variance_fraction=float(eig[-1]/eig.sum())))
(R/'unchanged_L_redundancy.json').write_text(json.dumps(latent,indent=2))
