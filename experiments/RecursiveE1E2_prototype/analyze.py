import json
import numpy as np,pandas as pd
from settings import ROOT,OUT

def main():
 frames=[];dec=[];checks=[]
 for suffix,cohort in [('', 'primary'),('_positive','positive_control')]:
  a=pd.read_csv(OUT/f'trajectory_metrics{suffix}.csv.gz');a['cohort']=cohort;a['system_index']=a.case.str[4:6].astype(int);frames.append(a);b=pd.read_csv(OUT/f'decisions{suffix}.csv');b['cohort']=cohort;b['system_index']=b.case.str[4:6].astype(int);dec.append(b);q=pd.read_csv(OUT/f'test_checks{suffix}.csv');q['cohort']=cohort;checks.append(q)
 t=pd.concat(frames,ignore_index=True);d=pd.concat(dec,ignore_index=True);q=pd.concat(checks,ignore_index=True);t.to_csv(OUT/'all_trajectory_metrics.csv.gz',index=False);d.to_csv(OUT/'all_decisions.csv',index=False);q.to_csv(OUT/'all_test_checks.csv',index=False);metrics=['NLL','CRPS','MSE','joint_cover90','marginal_cover90','width90','NLL_h1','NLL_h2','NLL_h5','NLL_h10'];t.groupby(['cohort','system_index','method'])[metrics].agg(['mean','std']).to_csv(OUT/'summary.csv');d.groupby(['cohort','system_index'])[['initial_state_supported','Q1','Q2','Q3','accept_new_coordinate']].sum().to_csv(OUT/'question_summary.csv');rng=np.random.default_rng(8102719);rows=[]
 for (cohort,index),group in t.groupby(['cohort','system_index']):
  for comparison in ['initial','long','direct2','direct_auto','full_X','full_W1','random','duplicate']:
   a=group[group.method=='recursive'].sort_values(['seed','trajectory']);b=group[group.method==comparison].sort_values(['seed','trajectory']);assert len(a)==300 and len(b)==300
   for metric in ['NLL','CRPS','MSE']:
    aa=a[metric].to_numpy().reshape(3,100);bb=b[metric].to_numpy().reshape(3,100);si=rng.integers(0,3,(2000,3));ti=rng.integers(0,100,(2000,3,100));v=(bb-aa)[si[:,:,None],ti].mean((1,2));lo,hi=np.quantile(v,[.025,.975]);rows.append(dict(cohort=cohort,system_index=int(index),comparison=comparison,metric=metric,gain=float((bb-aa).mean()),low95=float(lo),high95=float(hi),relative_reduction=float(1-aa.mean()/bb.mean()),independent_seeds=3,trajectories=300))
 pd.DataFrame(rows).to_csv(OUT/'paired_intervals.csv',index=False);strong=t[t.system_index==6].groupby('method')[metrics].mean();selected=d[(d.cohort=='positive_control')&(d.system_index==6)];result=dict(primary_cases=15,primary_new_coordinates_accepted=int(d[d.cohort=='primary'].accept_new_coordinate.sum()),strong_one_state_controls=3,strong_one_state_false_additions=int(d[d.system_index==5].accept_new_coordinate.sum()),strong_two_state_controls=3,strong_two_state_additions=int(selected.accept_new_coordinate.sum()),strong_two_state_NLL_initial=float(strong.loc['initial','NLL']),strong_two_state_NLL_recursive=float(strong.loc['recursive','NLL']),strong_two_state_CRPS_relative_reduction=float(1-strong.loc['recursive','CRPS']/strong.loc['initial','CRPS']),strong_two_state_MSE_relative_reduction=float(1-strong.loc['recursive','MSE']/strong.loc['initial','MSE']),main_scope='multi-step predictive-coordinate expansion from a restricted W1, not physical cause discovery or superiority over direct automatic dimensionality',one_step_warning='one-step NLL is almost unchanged; gains mainly joint10-step/multi-horizon',E3_run=False)
 (OUT/'CONCLUSION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
