import os,json
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
rng=np.random.default_rng(320000)
def interval(v,alpha=.05,B=4000):
 vv=[]
 for _,g in v.groupby(['noise','cohort']):
  x=g.value.to_numpy();vv.append(x[rng.integers(0,len(x),(B,len(x)))].mean(1))
 return list(map(float,np.quantile(np.mean(vv,axis=0),[alpha/2,1-alpha/2])))
def main():
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');d['noise']=d.noise.round(6);cur=pd.read_csv(OUT/'time_curves.csv');cur['noise']=cur.noise.round(6);summary=[]
 for (family,arm),g in d.groupby(['family','arm']):
  r=dict(family=family,arm=arm,n=len(g),extra_channels=int(arm!='X'),event_recall=float(g.detected.mean()),event_early_alarm=float(g.early_alarm.mean()),conditional_event_delay=float(g.delay.mean()),penalized_event_delay=float(g.delay.fillna(15).mean()),conditional_correct_ID_delay=float(g.id_delay.mean()),correct_first_commit=float(g.correct_commit.mean()),wrong_first_commit=float(g.wrong_commit.mean()),abstain=float(g.abstain.mean()),precision_among_committed=float(g.correct_commit.sum()/max(1,(1-g.abstain).sum())))
  for m in ['detect_by_1.0','detect_by_2.0','detect_by_5.0','correct_id_by_1.0','correct_id_by_2.0','correct_id_by_5.0','event_cover_0.5','event_cover_1.0','event_cover_2.0','id_cover_0.5','id_cover_1.0','id_cover_2.0']:r[m]=float(g[m].mean())
  summary.append(r)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
 contrasts=[];family=d[d.family=='four_mechanisms']
 for metric in ['detect_by_2.0','correct_id_by_2.0']:
  for arm in ['X_U','X_Y']:
   wide=family.pivot(index=['noise','cohort','trajectory','kappa','truth'],columns='arm',values=metric);delta=(wide[arm]-wide['X']).rename('value').reset_index();v=delta.groupby(['noise','cohort','trajectory']).value.mean().reset_index();ci=interval(v,.05/4);co=delta.groupby('cohort').value.mean().to_dict();mean=float(delta.value.mean());contrasts.append(dict(metric=metric,arm=arm,gain=mean,simultaneous_ci=ci,cohort_gains=co,passes=bool(mean>=.05 and ci[0]>0 and min(co.values())>0)))
 (OUT/'primary_contrasts.json').write_text(json.dumps(contrasts,indent=2))
 metrics=['detected','delay','correct_commit','wrong_commit','abstain','detect_by_2.0','correct_id_by_2.0','event_cover_0.5','event_cover_1.0','event_cover_2.0','id_cover_0.5','id_cover_1.0','id_cover_2.0']
 for cols,name in [(['family','arm','noise'],'by_noise'),(['family','arm','kappa'],'by_lag'),(['family','arm','truth'],'by_mechanism'),(['family','arm','cohort'],'by_cohort')]:d.groupby(cols)[metrics].mean().to_csv(OUT/(name+'.csv'))
 cur.groupby(['family','arm','time']).mean(numeric_only=True).to_csv(OUT/'curves_summary.csv')
 settings=json.loads((OUT/'trained_settings.json').read_text());audit=[]
 for ni,s in enumerate(settings):
  for arm,r in s['null_audit'].items():
   n=r['n'];p=r['fpr'];z=1.96;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;audit.append(dict(noise=SIGMAS[ni],arm=arm,fpr=p,count=r['count'],n=n,ci95=[ctr-half,ctr+half]))
 (OUT/'null_audit.json').write_text(json.dumps(audit,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 colors={'X':'#777777','X_U':'#167d9a','X_Y':'#d47a17'}
 for ax,metric,title in zip(axes,['event_coverage','correct_first_commit_coverage','instant_reject'],['First change alarm','Correct first mechanism decision','Instantaneous mechanism abstention']):
  for arm,g in cur[cur.family=='four_mechanisms'].groupby('arm'):
   z=g.groupby('time')[metric].mean();ax.plot(z.index,100*z,marker='o',label=arm,color=colors[arm])
  ax.axvline(10,color='black',linestyle=':',alpha=.4);ax.set_ylim(-2,102);ax.set_xlabel('Data time (source starts at 10)');ax.set_ylabel('Trajectories (%)');ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'online_curves.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9,4));ss=pd.DataFrame(summary);z=ss[ss.family=='four_mechanisms'].set_index('arm').loc[['X','X_U','X_Y']]
 for ax,prefix,title in zip(axes,['event_cover_','id_cover_'],['Change alarm before prediction-reference alarm','Correct first ID before prediction-reference alarm']):
  for j,margin in enumerate([.5,1.,2.]):ax.bar(np.arange(3)+(j-1)*.23,100*z[prefix+str(margin)],.23,label=f'lead >= {margin}')
  ax.set_xticks(range(3),z.index);ax.set_ylim(0,105);ax.set_title(title,fontsize=9);ax.set_ylabel('All-event coverage (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'lead_coverage.png',dpi=180);plt.close(fig)
 print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
