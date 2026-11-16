import os,json
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
rng=np.random.default_rng(720000)
def paired_interval(df,alpha=.0125,B=4000):
 vals=[]
 for _,g in df.groupby(['noise','plant','cohort']):
  a=g.value.to_numpy();vals.append(a[rng.integers(0,len(a),(B,len(a)))].mean(1))
 return list(map(float,np.quantile(np.mean(vals,axis=0),[alpha/2,1-alpha/2])))

def normal_audit(settings):
 rows=[];flags={}
 for ni in range(3):
  for ki in range(4):
   for pi,p in enumerate(POLICIES):
    cell=settings[f'n{ni}_k{ki}_p{pi}'];z=np.load(OUT/f'normal_n{ni}_k{ki}_p{pi}.npz')
    for budget in BUDGETS:
     for m,t in cell['thresholds'][str(budget)].items():
      f=z['null_'+m]>t;flags[ni,ki,p,m,budget]=f;v=f.mean();n=len(f);q=1.96;den=1+q*q/n;c=(v+q*q/(2*n))/den;h=q*np.sqrt(v*(1-v)/n+q*q/(4*n*n))/den;rows.append(dict(noise=SIGMAS[ni],plant=ki,tau_truth=KTEST[ki],policy=p,method=m,budget=budget,n=n,count=int(f.sum()),fpr=float(v),ci95_low=c-h,ci95_high=c+h))
 pd.DataFrame(rows).to_csv(OUT/'normal_audit.csv',index=False);return flags

def fairness(flags,p1,m1,p0,m0,budget=.05):
 diffs=[];v1=[];v0=[];vars=[]
 for ni in range(3):
  for ki in range(4):
   a=flags[ni,ki,p1,m1,budget].astype(float);b=flags[ni,ki,p0,m0,budget].astype(float);x=a-b;diffs.append(x.mean());v1.append(a.mean());v0.append(b.mean());vars.append(x.var(ddof=1)/len(x))
 delta=float(np.mean(diffs));se=np.sqrt(sum(vars))/len(vars);ci=[delta-1.96*se,delta+1.96*se]
 return dict(advantaged_FPR=float(np.mean(v1)),comparator_FPR=float(np.mean(v0)),difference=delta,paired_ci95=ci,passes=not(delta>.01 and ci[0]>0))

def main():
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');d['noise']=d.noise.round(6);d['budget']=d.budget.round(6);settings=json.loads((OUT/'settings.json').read_text());flags=normal_audit(settings);summary=[]
 for (mode,policy,method,budget),g in d.groupby(['mode','policy','method','budget']):
  r=dict(mode=mode,policy=policy,method=method,budget=budget,n=len(g),recall=float(g.recall.mean()),early_alarm=float(g.early_alarm.mean()),no_alarm=float(g.no_alarm.mean()),conditional_delay=float(g.delay.mean()) if g.delay.notna().any() else None,capped_delay=float(g.capped_delay.mean()),conditional_lead=float(g.lead.mean()) if g.lead.notna().any() else None,reference_valid=float(g.reference_valid.mean()),alarm_any=float((1-g.no_alarm).mean()))
  for margin in [.5,1.,2.]:r[f'cover_{margin}']=float(g[f'cover_{margin}'].mean())
  for delay in [1.,2.,4.]:r[f'recall_by_{delay}']=float(((g.recall==1)&(g.delay<=delay)).mean())
  summary.append(r)
 out=pd.DataFrame(summary);out.to_csv(OUT/'summary_all_budgets.csv',index=False);out[np.isclose(out.budget,.05)].to_json(OUT/'summary_main.json',orient='records',indent=2)
 contrasts=[]
 for mode in PRIMARY_MODES:
  main=d[(d['mode']==mode)&np.isclose(d.budget,.05)]
  for label,metric,p1,m1,p0,m0,sign,practical in [('same_input_model','cover_1.0','square','GainGLR','square','GLR',1,.05),('active_input_delay','capped_delay','square','GainGLR','none','GainGLR',-1,.25)]:
   a=main[(main.policy==p1)&(main.method==m1)];b=main[(main.policy==p0)&(main.method==m0)];keys=['noise','plant','cohort','trajectory'];pairs=a[keys+[metric]].merge(b[keys+[metric]],on=keys,suffixes=['_a','_b']);pairs['value']=sign*(pairs[metric+'_a']-pairs[metric+'_b']);ci=paired_interval(pairs);delta=float(pairs.value.mean());co=pairs.groupby('cohort').value.mean().to_dict();gate=fairness(flags,p1,m1,p0,m0);contrasts.append(dict(mode=mode,test=label,metric=metric,advantaged=f'{p1}/{m1}',comparator=f'{p0}/{m0}',improvement=delta,simultaneous_ci=ci,cohort_improvements=co,practical=practical,fairness_gate=gate,passes=bool(delta>=practical and ci[0]>0 and min(co.values())>0 and gate['passes'])))
 (OUT/'primary_contrasts.json').write_text(json.dumps(contrasts,indent=2))
 metrics=['recall','early_alarm','no_alarm','delay','capped_delay','cover_0.5','cover_1.0','cover_2.0','reference_valid']
 main=d[np.isclose(d.budget,.05)]
 for cols,name in [(['mode','policy','method','noise'],'by_noise'),(['mode','policy','method','tau_truth'],'by_tau'),(['mode','policy','method','cohort'],'by_cohort')]:main.groupby(cols)[metrics].mean().to_csv(OUT/(name+'.csv'))
 # Operational diagnostic: source-only changes have no causal-gain failure.
 source=out[(out['mode']=='source_only')&np.isclose(out.budget,.05)];source.to_csv(OUT/'source_only_specificity.csv',index=False)
 # Curves based on every trajectory; no selection on successful detection.
 headlines=[('none','CUSUM'),('none','GLR'),('none','M3_adapted'),('square','GLR'),('square','GainGLR')];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,mode in zip(axes,PRIMARY_MODES):
  for policy,m in headlines:
   g=main[(main['mode']==mode)&(main.policy==policy)&(main.method==m)];xx=np.arange(0,10.01,.5);yy=[np.mean((g.recall==1)&(g.delay<=t)) for t in xx];ax.plot(xx,100*np.array(yy),label=policy+'/'+m)
  ax.set_xlabel('Time after true beta-loss onset');ax.set_ylabel('All-event valid recall (%)');ax.set_ylim(0,102);ax.set_title(mode);ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(OUT/'detection_curves.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,mode in zip(axes,PRIMARY_MODES):
  g=out[(out['mode']==mode)&np.isclose(out.budget,.05)].set_index(['policy','method']);x=np.arange(len(headlines));width=.24
  for j,lead in enumerate([.5,1.,2.]):ax.bar(x+(j-1)*width,[100*g.loc[(p,m),f'cover_{lead}'] for p,m in headlines],width,label=f'lead >= {lead}')
  ax.set_xticks(x,[p+'/'+m for p,m in headlines],rotation=25,ha='right');ax.set_ylim(0,102);ax.set_ylabel('All-event coverage (%)');ax.set_title(mode+' / before own prediction alarm');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'lead_coverage.png',dpi=180);plt.close(fig)
 print(out[(out['mode'].isin(PRIMARY_MODES))&np.isclose(out.budget,.05)].to_string(index=False));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
