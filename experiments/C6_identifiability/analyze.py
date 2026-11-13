import os,json
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *

rng=np.random.default_rng(216000)
def interval(block,alpha=.05,B=4000):
 cols=[]
 for _,g in block.groupby(['noise','cohort']):
  v=g['value'].to_numpy();cols.append(v[rng.integers(0,len(v),(B,len(v)))].mean(1))
 vals=np.mean(cols,axis=0);return list(map(float,np.quantile(vals,[alpha/2,1-alpha/2])))
def aggci(d,value='correct'):
 block=d.groupby(['noise','cohort','trajectory'])[value].mean().reset_index(name='value');return interval(block)
def load(name):
 p=OUT/(name+'.json')
 return pd.read_json(p) if p.exists() else pd.read_csv(OUT/(name+'.csv.gz'))
def main():
 d=load('classifications');w=load('warnings');lat=load('latent_estimates');summ=[]
 d['noise']=d.noise.round(6);w['noise']=w.noise.round(6);lat['noise']=lat.noise.round(6)
 d['credible_contains_truth']=False
 for family,g in d.groupby('family'):
  nc=int(g.truth.max())+1;pp=g[[f'p{i}' for i in range(nc)]].to_numpy();order=np.argsort(-pp,axis=1);included=order<0
  for row in range(len(g)):included[row,:int(g.iloc[row].credible_size)]=True
  hits=np.any((order==g.truth.to_numpy()[:,None])&included,axis=1);d.loc[g.index,'credible_contains_truth']=hits
 for (family,obs),g in d.groupby(['family','observation']):
  selective=g[g.abstain==0]
  summ.append(dict(family=family,observation=obs,n=len(g),accuracy=float(g.correct.mean()),accuracy_ci95=aggci(g),abstention=float(g.abstain.mean()),selective_accuracy=None if not len(selective) else float(selective.correct.mean()),mean_credible_size=float(g.credible_size.mean()),credible_coverage=float(g.credible_contains_truth.mean()),median_true_log_odds=float(g.true_log_odds.median())))
 summary=pd.DataFrame(summ);summary.to_json(OUT/'summary.json',orient='records',indent=2)
 for cols,name in [(['family','observation','noise'],'by_noise'),(['family','observation','kappa'],'by_lag'),(['family','observation','cohort'],'by_cohort')]:d.groupby(cols).agg(accuracy=('correct','mean'),abstention=('abstain','mean'),n=('correct','size')).to_csv(OUT/(name+'.csv'))
 d.groupby(['family','observation','truth','prediction']).size().rename('count').to_csv(OUT/'confusion.csv')
 gains=[]
 comparisons=[('main_passive','X',f'X_VU_{s}') for s in SENSOR]+[('causal_equivalent','X',f'X_VY_{s}') for s in SENSOR]+[('main_active','X_input_hidden','X_input_known'),('causal_active','X_input_hidden','X_input_known')]
 for family,baseline,added in comparisons:
  g=d[(d.family==family)&d.observation.isin([baseline,added])];pv=g.pivot(index=['noise','cohort','trajectory','kappa','truth'],columns='observation',values='correct');delta=(pv[added]-pv[baseline]).rename('value').reset_index();block=delta.groupby(['noise','cohort','trajectory']).value.mean().reset_index();ci=interval(block,.05/8);co=delta.groupby('cohort').value.mean().to_dict();mean=float(delta.value.mean());gains.append(dict(family=family,baseline=baseline,added=added,gain=mean,simultaneous_ci=ci,cohort_gains=co,passes=bool(mean>=.05 and ci[0]>0 and min(co.values())>0)))
 (OUT/'primary_gains.json').write_text(json.dumps(gains,indent=2))
 ws=w.groupby('method').agg(n=('detected','size'),recall=('detected','mean'),early_FPR=('early_false_alarm','mean'),delay=('delay','mean'),lead_conditional=('lead','mean'),cover_025=('cover_0.25','mean'),cover_05=('cover_0.5','mean'),cover_1=('cover_1.0','mean'));ws.to_csv(OUT/'warning_summary.csv')
 lat.groupby('truth').agg(onset_MAE=('onset_error','mean'),onset_median_error=('onset_error','median'),n=('onset_error','size')).to_csv(OUT/'onset_summary.csv')
 # compressed row-level outputs, retain all classifications and warnings.
 for name,df in [('classifications',d),('warnings',w),('latent_estimates',lat)]:df.to_csv(OUT/(name+'.csv.gz'),index=False,compression='gzip')
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for ax,family,obs,title in zip(axes,['main_passive','causal_equivalent','main_active'],[['X']+[f'X_VU_{s}' for s in SENSOR],['X','X_VU_0.1']+[f'X_VY_{s}' for s in SENSOR],['X_input_hidden','X_input_known']],['Four mechanisms: extra force sensor','Passive-equivalent causal pair','Four mechanisms: source intervention']):
  z=summary[(summary.family==family)].set_index('observation').loc[obs];ax.bar(range(len(z)),100*z.accuracy,color=['#777777']+['#197b91']*(len(z)-1));ax.set_xticks(range(len(z)),[v.replace('X_','').replace('input_','I ').replace('_',' ') for v in obs],rotation=25,ha='right');ax.set_ylim(0,105);ax.set_title(title,fontsize=10);ax.set_ylabel('Mechanism classification accuracy (%)');ax.grid(axis='y',alpha=.2)
  for i,v in enumerate(z.accuracy):ax.text(i,100*v+1,f'{100*v:.1f}',ha='center',fontsize=9)
 fig.tight_layout();fig.savefig(OUT/'information_gain.png',dpi=180);fig.savefig(OUT/'information_gain.svg');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,family,obs in zip(axes,['main_passive','main_passive','main_active'],['X','X_VU_0.1','X_input_known']):
  g=d[(d.family==family)&(d.observation==obs)];ct=pd.crosstab(g.truth,g.prediction).reindex(index=range(4),columns=range(4),fill_value=0);mat=ct.to_numpy()/ct.sum(axis=1).to_numpy()[:,None];im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),['A','B','C','D']);ax.set_yticks(range(4),['A','B','C','D']);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(family+' / '+obs,fontsize=9)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{mat[i,j]*100:.0f}',ha='center',va='center',color='white' if mat[i,j]>.55 else 'black')
 fig.tight_layout();fig.savefig(OUT/'confusion.png',dpi=180);plt.close(fig)
 print(summary.to_string(index=False));print(json.dumps(gains,indent=2));print(ws.to_string())
if __name__=='__main__':main()
