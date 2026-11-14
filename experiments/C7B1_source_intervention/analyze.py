import os,json
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
rng=np.random.default_rng(421000)
def interval(df,alpha=.05,B=4000):
 vals=[]
 for _,g in df.groupby(['noise','cohort']):
  x=g.value.to_numpy();vals.append(x[rng.integers(0,len(x),(B,len(x)))].mean(1))
 return list(map(float,np.quantile(np.mean(vals,axis=0),[alpha/2,1-alpha/2])))
def main():
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');d['noise']=d.noise.round(6);cur=pd.read_csv(OUT/'time_curves.csv');cur['noise']=cur.noise.round(6);summary=[]
 for (key,arm),g in d.groupby(['input','arm']):
  block=g.groupby(['noise','cohort','trajectory']).correct_first.mean().reset_index(name='value');r=dict(input=key,arm=arm,n=len(g),correct_first=float(g.correct_first.mean()),wrong_first=float(g.wrong_first.mean()),refuse=float(g.refuse.mean()),hard_final_accuracy=float(g.final_hard_accuracy.mean()),conditional_correct_delay=None if not g.delay.notna().any() else float(g.delay.mean()),penalized_delay=float(g.delay.fillna(15).mean()),mean_energy=float(g.energy_total.mean()),mean_energy_at_any_commit=None if not g.energy_at_commit.notna().any() else float(g.energy_at_commit.mean()),nominal_false_anomaly=None if not g.nominal_false_anomaly.notna().any() else float(g.nominal_false_anomaly.mean()),correct_first_ci95=interval(block))
  for h in [.5,1.,2.,4.,8.,15.]:r[f'correct_by_{h}']=float(g[f'correct_by_{h}'].mean())
  summary.append(r)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
 comparisons=[]
 for key in INPUTS[1:]:
  sub=d[(d.arm=='X')&d.input.isin(['I0_passive',key])];pv=sub.pivot(index=['noise','cohort','trajectory','kappa','truth'],columns='input',values='correct_by_4.0');diff=(pv[key]-pv.I0_passive).rename('value').reset_index();block=diff.groupby(['noise','cohort','trajectory']).value.mean().reset_index();ci=interval(block,.05/3);co=diff.groupby('cohort').value.mean().to_dict();mean=float(diff.value.mean());comparisons.append(dict(input=key,metric='X_correct_first_by14',gain=mean,simultaneous_ci=ci,cohort_gains=co,passes=bool(mean>=.05 and ci[0]>0 and min(co.values())>0)))
 (OUT/'primary_contrasts.json').write_text(json.dumps(comparisons,indent=2))
 metrics=['correct_first','wrong_first','refuse','final_hard_accuracy','delay','correct_by_2.0','correct_by_4.0','correct_by_15.0','nominal_false_anomaly']
 for cols,name in [(['input','arm','noise'],'by_noise'),(['input','arm','kappa'],'by_tau'),(['input','arm','truth'],'by_identity'),(['input','arm','cohort'],'by_cohort')]:d.groupby(cols)[metrics].mean().to_csv(OUT/(name+'.csv'))
 cur.groupby(['input','arm','time']).mean(numeric_only=True).to_csv(OUT/'curves_summary.csv')
 audit=[]
 for ni,ss in enumerate(json.loads((OUT/'trained_settings.json').read_text())):
  for arm,r in ss['nominal_audit'].items():
   p=r['fpr'];n=r['n'];z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;audit.append(dict(noise=SIGMAS[ni],arm=arm,nominal_fpr=p,ci95=[center-half,center+half],by_input=r['by_input'],blank_false_identity_commits=r['blank_false_identity_commits']))
 (OUT/'nominal_audit.json').write_text(json.dumps(audit,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(11,4));colors={'I0_passive':'#777777','I1_step_hold':'#177f91','I2_pulse':'#e39135','I3_random':'#7b56a5'}
 for ax,arm in zip(axes,ARMS):
  for key,g in cur[cur.arm==arm].groupby('input'):
   zz=g.groupby('time').correct_first_coverage.mean();ax.plot(zz.index,100*zz,marker='o',label=key,color=colors[key])
  ax.axvspan(10,14,color='grey',alpha=.08);ax.set_ylim(-2,102);ax.set_title('Correct first identity decision / '+arm);ax.set_xlabel('Time (probe starts at10)');ax.set_ylabel('All-trial coverage (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'online_identification.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));zz=pd.DataFrame(summary);zz=zz[zz.arm=='X'].set_index('input').loc[INPUTS];ax.bar(INPUTS,100*zz.correct_first,label='Correct first');ax.bar(INPUTS,100*zz.wrong_first,bottom=100*zz.correct_first,label='Wrong first');ax.bar(INPUTS,100*zz.refuse,bottom=100*(zz.correct_first+zz.wrong_first),label='Refusal');ax.set_ylim(0,105);ax.set_ylabel('All trials (%)');ax.set_title('X-only first-decision outcomes through25');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'first_decisions.png',dpi=180);plt.close(fig)
 print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
