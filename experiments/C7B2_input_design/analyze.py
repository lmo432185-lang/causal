import os,json
os.environ['MPLCONFIGDIR']='/workspace/work/mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
from physics import command
from run import summary,pareto
rng=np.random.default_rng(620000)
def interval(df,alpha=.0125,B=4000):
 vals=[]
 for _,g in df.groupby(['noise','cohort']):
  x=g.value.to_numpy();vals.append(x[rng.integers(0,len(x),(B,len(x)))].mean(1))
 return list(map(float,np.quantile(np.mean(vals,axis=0),[alpha/2,1-alpha/2])))
def main():
 d=pd.read_csv(OUT/'test_metrics.csv.gz');d['noise']=d.noise.round(6);s=pd.read_csv(OUT/'stress_metrics.csv.gz');s['noise']=s.noise.round(6);sel=json.loads((OUT/'SELECTION.json').read_text());tab=summary(d);tab.to_csv(OUT/'test_summary.csv',index=False);roles=dict(reference=REFERENCE,**sel['policies']);out=[]
 for role,name in roles.items():
  if name is None:continue
  g=d[d.program==name];r=tab.set_index('program').loc[name].to_dict();r.update(role=role,program=name,correct=float(g.correct.mean()),wrong=float(g.wrong.mean()),refuse=float(g.refuse.mean()),conditional_delay=float(g.delay.mean()),final_hard_accuracy=float(g.final_hard_correct.mean()),energy_at_commit=float(g.energy_at_commit.mean()),nominal_fpr=float(g.nominal_false_anomaly.mean()))
  for h in [.5,1.,2.,4.,8.,15.]:r[f'cover_{h}']=float(g[f'cover_{h}'].mean())
  out.append(r)
 (OUT/'policy_results.json').write_text(json.dumps(out,indent=2));pd.DataFrame(out).to_csv(OUT/'policy_results.csv',index=False)
 contrasts=[]
 for label,role,metric,sign,margin,practical in [('J_superiority','bestJ','J',-1,0,.02),('coverage_noninferiority','bestJ','cover_4.0',1,-.05,None),('energy_noninferiority','lowest_energy','cover_4.0',1,-.10,None),('high_noise_superiority','high_noise','cover_4.0',1,0,.05)]:
  name=roles[role]
  if name is None:contrasts.append(dict(test=label,available=False));continue
  data=d if label!='high_noise_superiority' else d[np.isclose(d.noise,.7)];pv=data[data.program.isin({REFERENCE,name})].pivot(index=['noise','cohort','trajectory','kappa','truth'],columns='program',values=metric);delta=(sign*(pv[name]-pv[REFERENCE])).rename('value').reset_index();block=delta.groupby(['noise','cohort','trajectory']).value.mean().reset_index();ci=interval(block);mean=float(delta.value.mean());co=delta.groupby('cohort').value.mean().to_dict();energy=float(tab.set_index('program').loc[name,'energy']);ok=ci[0]>margin and min(co.values())>margin
  if practical is not None:ok=ok and mean>=practical
  if label=='energy_noninferiority':ok=ok and energy<4-1e-9
  contrasts.append(dict(test=label,available=True,selected=name,metric=metric,mean_difference=mean,simultaneous_ci=ci,cohort_differences=co,margin=margin,practical=practical,energy=energy,passes=bool(ok)))
 (OUT/'primary_contrasts.json').write_text(json.dumps(contrasts,indent=2))
 metrics=['J','energy','peak','correct','wrong','refuse','tcap','delay','cover_4.0','final_hard_correct','nominal_false_anomaly']
 for data,prefix in [(d,'test'),(s,'stress')]:
  for cols,suffix in [(['program','noise'],'noise'),(['program','kappa'],'tau'),(['program','cohort'],'cohort'),(['program','truth'],'identity')]:data.groupby(cols)[metrics].mean().to_csv(OUT/f'{prefix}_by_{suffix}.csv')
 audit=[]
 for ni,ss in enumerate(json.loads((OUT/'trained_settings.json').read_text())):
  for name,r in ss['nominal_audit'].items():
   p=r['fpr'];n=r['n'];z=1.96;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;audit.append(dict(program=name,noise=SIGMAS[ni],n=n,count=r['count'],fpr=p,ci95_low=ctr-half,ci95_high=ctr+half))
 pd.DataFrame(audit).to_csv(OUT/'nominal_audit.csv',index=False)
 (OUT/'test_pareto_descriptive.json').write_text(json.dumps(pareto(tab[tab.program.isin(PRIMARY)]),indent=2))
 # Duration diagnostic: same high cap for all five, marked infeasible for primary cap.
 names=[f'pulse_d{t:g}_E4' for t in [.1,.25,.5,1.,2.]];zz=d[d.program.isin(names)].groupby('program').agg(coverage=('cover_4.0','mean'),refuse=('refuse','mean'),wrong=('wrong','mean'),peak=('peak','max')).reindex(names);zz.to_csv(OUT/'duration_diagnostic.csv')
 fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar([.1,.25,.5,1.,2.],100*zz.coverage.to_numpy(),width=[.07,.1,.18,.3,.45],color=['#be6559','#be6559','#167e99','#167e99','#167e99']);ax.set_xticks([.1,.25,.5,1.,2.]);ax.set_xlabel('Pulse duration');ax.set_ylabel('Correct first decision by14 (%)');ax.set_title('E=4 duration diagnostic; common cap=sqrt(40)\nRed: infeasible under primary cap=sqrt(8)',fontsize=10);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'duration_diagnostic.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4));colors={'reference':'#777777','bestJ':'#177f91','fastest':'#e39135','lowest_energy':'#7b56a5','high_noise':'#4b8b4a'}
 grouped={}
 for role,name in roles.items():
  if name is not None:grouped.setdefault(name,[]).append(role)
 for name,rolelist in grouped.items():
  role=rolelist[0];label='/'.join(rolelist);g=d[d.program==name];covers=[g[f'cover_{h}'].mean() for h in [.5,1.,2.,4.,8.,15.]];axes[0].plot([.5,1,2,4,8,15],100*np.array(covers),marker='o',label=label,color=colors[role]);row=tab.set_index('program').loc[name];axes[1].scatter(row.energy,100*row.cover,label=label,color=colors[role],s=80);axes[1].annotate(label,(row.energy,100*row.cover),xytext=(3,4),textcoords='offset points',fontsize=8)
 axes[0].set_xlabel('Time since intervention');axes[0].set_ylabel('Correct first coverage (%)');axes[0].set_title('Validation-frozen policies on new test');axes[0].legend(fontsize=8);axes[1].set_xlabel('Planned total energy');axes[1].set_ylabel('Correct first coverage by14 (%)');axes[1].set_title('Cost vs coverage; no test reselection');axes[1].set_xlim(1.8,4.9);axes[0].set_ylim(0,65);axes[1].set_ylim(0,65)
 for a in axes:a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'frozen_policies.png',dpi=180);plt.close(fig)
 # Full waveform and energy accounting of fixed selected programs.
 fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
 for name in sorted({v for v in roles.values() if v is not None}):
  I=command(1,name,SEEDS['test'][0][0])[0];t=np.arange(T)*DT;axes[0].step(t,I,where='post',label=name);axes[1].plot((np.arange(T)+1)*DT,np.cumsum(I*I)*DT,label=name)
 axes[0].set_ylabel('Source command');axes[1].set_ylabel('Cumulative energy');axes[1].set_xlabel('Time');axes[1].set_xlim(9.75,14.25);axes[0].legend(fontsize=7,ncol=2);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'commands_and_energy.png',dpi=180);plt.close(fig)
 val=pd.read_csv(OUT/'validation_summary.csv');val=val[val.program.isin(PRIMARY)];front=val[val.program.isin(sel['pareto'])];fig,ax=plt.subplots(figsize=(8,4));sc=ax.scatter(val.energy,100*val.cover,c=val.peak,cmap='viridis',s=45);ax.scatter(front.energy,100*front.cover,facecolors='none',edgecolors='black',s=110,label='5D validation Pareto set');ax.set_xlim(.7,4.9);ax.set_ylim(0,65);ax.set_xlabel('Planned total energy');ax.set_ylabel('Validation correct-first coverage by14 (%)');ax.set_title('44 feasible candidates; no test-driven selection');ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Actual peak');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'validation_pareto.png',dpi=180);plt.close(fig)
 print(pd.DataFrame(out).to_string(index=False));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
