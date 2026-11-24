"""Registered main comparisons; failed model availability is never hidden."""
import json,time,numpy as np,pandas as pd
from scipy.stats import ttest_1samp,nct,t
from settings import *
from confirm import holm
MAIN={'residual_gain':{'R0':1,'R2':-1},'joint_gain':{'R0':1,'R3':-1},'shared_vs_old_mix':{'J0':1,'J1':-1},'posterior_vs_point':{'J1':1,'J2':-1},'feedback_gain':{'C':1,'D':-1},'complement_gain':{'BestSingle':1,'E':-1}}
MECHANISM={'original_vs_residual':{'R1_original':1,'R2':-1},'original_vs_joint':{'R1_original':1,'R3':-1},'path_vs_residual':{'R1':1,'R2':-1}}
SECOND={'dense_parameter_gain':{'J1':1,'J1_param':-1},'structure_gain':{'J1_param':1,'J2':-1},'parameter_with_structures':{'J2_point':1,'J2':-1},'full_direct_reference':{'StrongestSingle':1,'E':-1},'online_update':{'M1':1,'M3':-1}}

def ci(d,value='delta',hier=True,seed=5217):
 blocks=[g[value].dropna().to_numpy() for _,g in d.groupby('seed',sort=True)];blocks=[b for b in blocks if len(b)];n=len(blocks)
 if n<2:return np.nan,np.nan,np.nan,np.nan
 rng=np.random.default_rng(seed);means=np.array([b.mean() for b in blocks]);chosen=rng.integers(0,n,(10000,n));samples=np.empty_like(chosen,dtype=float)
 if hier:
  for i,b in enumerate(blocks):mask=chosen==i;samples[mask]=rng.choice(b,size=(int(mask.sum()),len(b)),replace=True).mean(1)
 else:samples=means[chosen]
 vals=samples.mean(1);p=float(ttest_1samp(means,.005,alternative='greater').pvalue);return float(means.mean()),*np.quantile(vals,[.025,.975]).tolist(),p

def compact_interval(means):
 a=np.array(means,dtype=float);a=a[np.isfinite(a)]
 if len(a)<2:return (np.nan,np.nan,np.nan)
 rng=np.random.default_rng(61573);draw=rng.choice(a,size=(10000,len(a)),replace=True).mean(1);return float(a.mean()),*np.quantile(draw,[.025,.975]).tolist()

def main():
 rows=[];data=[]
 for batch in ['A','B']:
  d=pd.read_csv(OUT/f'test_{batch}.csv.gz');d['condition']=d.case.str.split('_').str[0];data.append(d);p=d.pivot(index=['condition','case','seed','trajectory'],columns='method',values='NLL').reset_index()
  for name,coef in {**MAIN,**SECOND,**MECHANISM}.items():
   p['delta']=sum(p[k]*v for k,v in coef.items())
   for cond in ['ALL']+sorted(p.condition.unique()):
    g=p if cond=='ALL' else p[p.condition==cond];valid=g[np.isfinite(g.delta)];mean,lo,hi,pv=ci(valid);rows.append(dict(batch=batch,condition=cond,contrast=name,family='main' if name in MAIN else 'mechanism' if name in MECHANISM else 'secondary' if name!='online_update' else 'diagnostic',gain=mean,lo=lo,hi=hi,p=pv,total_pairs=len(g),valid_pairs=len(valid),failed_pairs=len(g)-len(valid),seeds=valid.seed.nunique()))
 # Independent capacity diagnostic family, paired over training replicas before seed inference.
 for batch in ['A','B']:
  controls=pd.read_csv(OUT/f'controls_{batch}.csv.gz');controls['condition']=controls.case.str.split('_').str[0];cp=controls[controls.available].pivot(index=['condition','case','seed','trajectory','bank'],columns='control',values='NLL').reset_index()
  for mode in ['obs','direct_latent','direct_both','direct_dense']:
   cp['delta']=cp['none']-cp[mode];z=cp.groupby(['condition','case','seed','trajectory']).delta.mean().reset_index()
   for cond in ['ALL']+sorted(z.condition.unique()):
    g=z if cond=='ALL' else z[z.condition==cond];v=g[np.isfinite(g.delta)];mean,lo,hi,pv=ci(v);rows.append(dict(batch=batch,condition=cond,contrast='capacity_'+mode,family='capacity',gain=mean,lo=lo,hi=hi,p=pv,total_pairs=len(g),valid_pairs=len(v),failed_pairs=len(g)-len(v),seeds=v.seed.nunique()))
 r=pd.DataFrame(rows);r['p_holm']=np.nan
 for (batch,family),g in r.groupby(['batch','family']):
  if family=='diagnostic':continue
  for all_cond in [True,False]:ix=g.index[(g.condition=='ALL')==all_cond];r.loc[ix,'p_holm']=holm(r.loc[ix,'p'])
 r['practical_success']=(r.gain>=.005)&(r.lo>0)&(r.p_holm<.05)&(r.failed_pairs==0);r.to_csv(OUT/'contrasts.csv',index=False);d=pd.concat(data);d.groupby(['batch','condition','method']).agg(NLL=('NLL','mean'),CRPS=('CRPS','mean'),cover90=('cover90','mean'),width90=('width90','mean'),MSE=('MSE','mean'),ESS=('particle_ESS','mean'),entropy=('model_entropy','mean'),available=('available','mean')).reset_index().to_csv(OUT/'prediction_summary.csv',index=False)
 s=pd.read_csv(OUT/'confirmation_support.csv');s['condition']=s.case.str.split('_').str[0];s.groupby(['condition','route']).agg(support=('supported','sum'),n=('supported','size'),functional_match=('function_match','sum'),four_tests=('four_tests_passed','sum')).reset_index().to_csv(OUT/'support_summary.csv',index=False)
 # Physical probability summaries and seed-cluster confidence intervals.
 marg=pd.read_csv(OUT/'state_marginal.csv.gz');marg['condition']=marg.case.str.split('_').str[0];joint=pd.read_csv(OUT/'state_joint.csv.gz');joint['condition']=joint.case.str.split('_').str[0];cal=[]
 for (batch,cond,method,coordinate),g in marg.groupby(['batch','condition','method','coordinate']):
  for metric in ['NLL','CRPS','MSE','cover50','cover80','cover90','cover95','width50','width80','width90','width95']:
   v=g.groupby('seed')[metric].mean();mean,lo,hi=compact_interval(v);cal.append(dict(batch=batch,condition=cond,method=method,coordinate=coordinate,metric=metric,mean=mean,lo=lo,hi=hi,seeds=len(v)))
 for (batch,cond,method),g in joint[joint.available].groupby(['batch','condition','method']):
  for metric in ['joint_NLL','posterior_corr','joint_cover50','joint_cover80','joint_cover90','joint_cover95']:
   v=g.groupby('seed')[metric].mean();mean,lo,hi=compact_interval(v);cal.append(dict(batch=batch,condition=cond,method=method,coordinate='joint',metric=metric,mean=mean,lo=lo,hi=hi,seeds=len(v)))
 pd.DataFrame(cal).to_csv(OUT/'calibration_intervals.csv',index=False)
 # C16 primary and Bayesian C12 secondary diagnostics; same seed pairing.
 contrasts=[];mp=marg.groupby(['batch','condition','seed','method']).mean(numeric_only=True).reset_index();jp=joint[joint.available].groupby(['batch','condition','seed','method']).mean(numeric_only=True).reset_index()
 for batch in ['A','B']:
  for cond in ['CL','CM','CH','CE']:
   z=mp[(mp.batch==batch)&(mp.condition==cond)];j=jp[(jp.batch==batch)&(jp.condition==cond)];z=z.set_index(['seed','method']);j=j.set_index(['seed','method'])
   def get(method,metric,joint_metric=False):return (j if joint_metric else z).xs(method,level='method')[metric]
   tests=[('primary','fullQ_point_coverage',abs(get('INDEP_POINT','cover90')-.9)-abs(get('FULLQ_POINT','cover90')-.9)),('primary','parameter_coverage',abs(get('INDEP_POINT','cover90')-.9)-abs(get('PARAM_ONLY','cover90')-.9)),('primary','fullQ_CRPS',get('INDEP_POINT','CRPS')-get('FULLQ_PARAM','CRPS')),('primary','fullQ_joint_NLL',get('INDEP_POINT','joint_NLL',True)-get('FULLQ_PARAM','joint_NLL',True)),('bayes','continuous_parameter_coverage',abs(get('BAYES_POINT','cover90')-.9)-abs(get('BAYES_PARAM','cover90')-.9)),('bayes','continuous_fullQ_CRPS',get('BAYES_PARAM','CRPS')-get('BAYES_FULL','CRPS')),('bayes','continuous_fullQ_joint_NLL',get('BAYES_PARAM','joint_NLL',True)-get('BAYES_FULL','joint_NLL',True))]
   for family,name,v in tests:
    mean,lo,hi=compact_interval(v);pval=float(ttest_1samp(v,0,alternative='greater').pvalue);contrasts.append(dict(batch=batch,condition=cond,family=family,contrast=name,gain=mean,lo=lo,hi=hi,p=pval))
 cc=pd.DataFrame(contrasts);cc['p_holm']=np.nan
 for _,g in cc.groupby(['batch','family']):cc.loc[g.index,'p_holm']=holm(g.p)
 cc.to_csv(OUT/'calibration_contrasts.csv',index=False)
 sig=pd.read_csv(OUT/'signal_probability.csv.gz');sig['condition']=sig.case.str.split('_').str[0];sig.groupby(['batch','condition','method']).mean(numeric_only=True).reset_index().to_csv(OUT/'signal_summary.csv',index=False)
 jointx=pd.read_csv(OUT/'joint_X_probability.csv.gz');jointx['condition']=jointx.case.str.split('_').str[0];jxrows=[]
 for (batch,cond,method),g in jointx.groupby(['batch','condition','method']):
  for metric in ['joint_cover50','joint_cover80','joint_cover90','joint_cover95']:
   mean,lo,hi=compact_interval(g.groupby('seed')[metric].mean());jxrows.append(dict(batch=batch,condition=cond,method=method,metric=metric,mean=mean,lo=lo,hi=hi))
 pd.DataFrame(jxrows).to_csv(OUT/'joint_X_intervals.csv',index=False)
 power=[]
 for tests in [1,6,144]:
  critical=t.ppf(1-.05/tests,19)
  for sd in [.003,.0075,.015]:
   for true in [.005,.010,.015]:power.append(dict(n=20,comparisons=tests,sd=sd,true_gain=true,threshold=.005,power=float(nct.sf(critical,19,(true-.005)*np.sqrt(20)/sd))))
 pd.DataFrame(power).to_csv(OUT/'registered_power.csv',index=False);(ROOT/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),bootstrap=10000,no_refitting=True,primary_tests_per_batch=6,condition_tests_per_batch=144,C_primary=16,C_bayes_secondary=12),indent=2));print('Analysis complete',flush=True)
if __name__=='__main__':main()
