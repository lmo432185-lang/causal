import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent;OUT=ROOT/'results'
EDGES={'G1':{'YX','ZX'},'G2':{'YZ','ZX'},'G3':{'YX','YZ','ZX'},'G4':{'ZY','YX'}}
DIRECT={k:{v for v in vs if v.endswith('X')} for k,vs in EDGES.items()}
PATH={k:({'YZX'} if {'YZ','ZX'}<=v else set())|({'ZYX'} if {'ZY','YX'}<=v else set()) for k,v in EDGES.items()}
DIR={'G1':'none','G2':'YZ','G3':'YZ','G4':'ZY'}
FAM={'G1':'parallel','G2':'chain','G3':'mixed','G4':'chain'}

def main():
 assert (ROOT/'FINAL_EVALUATION_DONE.json').exists()
 if (ROOT/'positive_public_jobs.json').exists():assert (ROOT/'POSITIVE_EVALUATION_DONE.json').exists()
 private=json.loads((ROOT/'data/sealed_manifest.json').read_text())
 if (ROOT/'data/positive_sealed_manifest.json').exists():private+=json.loads((ROOT/'data/positive_sealed_manifest.json').read_text())
 rows=[];state=[];fit=[];case_rows=[]
 for meta in private:
  case=meta['case'];d=OUT/case;dec=json.loads((d/'X_only_decisions.json').read_text());orc=json.loads((d/'oracle_results.json').read_text());supp=json.loads((d/'supplementary_law_audit.json').read_text());post=np.array(dec['posterior']);truth=meta['truth'];best=dec['selected_by_validation'];f=json.loads((d/'fit_summary.json').read_text());fit+=f
  cp=dict(case=case,section=meta['section'],tag=meta['tag'],truth=truth,seed=meta['seed'],model_selected_by_validation=best,mirror_loglik_error=dec['mirror_loglik_error'],equivalent_alternative_counts={k:len(v) for k,v in dec['estimated_equivalent_alternatives'].items()});case_rows.append(cp)
  for i in range(200):
   pred=dec['prediction'][i];conf=float(post[i].max())
   for method,decl in [('forced',True),('naive_score',dec['naive_declared'][i]),('equivalence_guard',dec['guarded_declared'][i]),('supplementary_audit',supp['supplementary_guarded_declared'][i]),('oracle',True)]:
    pr=orc['predictions'][i] if method=='oracle' else pred;co=float(max(orc['posterior'][i])) if method=='oracle' else conf;tp=len(DIRECT[truth]&DIRECT[pr]) if decl else 0;fp=len(DIRECT[pr]-DIRECT[truth]) if decl else 0;fn=len(DIRECT[truth])-tp;rows.append(dict(case=case,section=meta['section'],tag=meta['tag'],seed=meta['seed'],trajectory=i,truth=truth,method=method,prediction=pr if decl else 'refuse',forced_prediction=pr,declared=int(decl),correct_declared=int(decl and pr==truth),wrong_declared=int(decl and pr!=truth),refused=int(not decl),score_confidence=co,direct_TP=tp,direct_FP=fp,direct_FN_all=fn,direction_correct=int(decl and DIR[pr]==DIR[truth]),indirect_path_correct=int(decl and PATH[pr]==PATH[truth]),family_correct=int(decl and FAM[pr]==FAM[truth]),single_vs_selected_loglik=float(np.array(dec['test_LL'])[i,['G1','G2','G3','G4'].index(best)]-dec['single_test_LL'][i])))
  s=pd.read_csv(d/'hidden_state_metrics.csv.gz');s['section']=meta['section'];s['tag']=meta['tag'];s['truth']=truth;s['seed']=meta['seed'];state.append(s)
 out=pd.DataFrame(rows);out.to_csv(OUT/'all_trajectory_results.csv.gz',index=False);ss=pd.concat(state,ignore_index=True);ss.to_csv(OUT/'all_hidden_state_metrics.csv.gz',index=False);pd.DataFrame(fit).to_csv(OUT/'optimizer_summary.csv',index=False);(OUT/'case_evaluation.json').write_text(json.dumps(case_rows,indent=2))
 def summary(g):
  n=len(g);dn=g.declared.sum();tp=g.direct_TP.sum();fp=g.direct_FP.sum();fn=g.direct_FN_all.sum();return pd.Series(dict(n=n,coverage=dn/n,correct_coverage=g.correct_declared.mean(),wrong_coverage=g.wrong_declared.mean(),refusal=g.refused.mean(),selective_accuracy=g.correct_declared.sum()/dn if dn else np.nan,direct_precision=tp/(tp+fp) if tp+fp else np.nan,direct_recall_all=tp/(tp+fn) if tp+fn else np.nan,direction_correct_all=g.direction_correct.mean(),indirect_path_correct_all=g.indirect_path_correct.mean(),family_correct_all=g.family_correct.mean()))
 for keys,name in [(['section','method'],'section'),(['section','tag','method'],'conditions'),(['section','tag','truth','method'],'truth_conditions'),(['section','tag','seed','method'],'seeds')]:out.groupby(keys).apply(summary,include_groups=False).to_csv(OUT/f'summary_{name}.csv')
 ss.groupby(['section','tag','family','mode','state'])[['MSE','NMSE','corr']].mean().to_csv(OUT/'hidden_state_summary.csv');pd.crosstab(out[(out.section=='A')&(out.method=='forced')].truth,out[(out.section=='A')&(out.method=='forced')].prediction,normalize='index').to_csv(OUT/'confusion_forced_A.csv');pd.crosstab(out[(out.section=='A')&(out.method=='equivalence_guard')].truth,out[(out.section=='A')&(out.method=='equivalence_guard')].prediction,normalize='index').to_csv(OUT/'confusion_guard_A.csv')
 # Reliability is evaluated after freeze, never used to fit a temperature or threshold.
 calib=[]
 for method in ['forced','oracle']:
  a=out[(out.section=='A')&(out.method==method)]
  for b in range(10):
   z=a[(a.score_confidence>=b/10)&(a.score_confidence<((b+1)/10 if b<9 else 1.0000001))]
   if len(z):calib.append(dict(method=method,bin=b,n=len(z),confidence=float(z.score_confidence.mean()),accuracy=float(z.correct_declared.mean())))
 pd.DataFrame(calib).to_csv(OUT/'reliability_A.csv',index=False);ece={k:sum(v['n']*abs(v['confidence']-v['accuracy']) for v in calib if v['method']==k)/sum(v['n'] for v in calib if v['method']==k) for k in ['forced','oracle']};(OUT/'calibration_ECE_A.json').write_text(json.dumps(ece,indent=2))
 # Conditional-on-fit intervals: bootstrap trajectories within each dataset.
 rng=np.random.default_rng(984000);interval=[]
 for (section,tag,method),a in out.groupby(['section','tag','method']):
  arr=np.stack([v.sort_values('trajectory').correct_declared.to_numpy() for _,v in a.groupby('case')]);bs=np.array([np.take_along_axis(arr,rng.integers(0,200,arr.shape),axis=1).mean() for _ in range(1000)]);interval.append(dict(section=section,tag=tag,method=method,correct_coverage=float(arr.mean()),conditional_CI95=np.quantile(bs,[.025,.975]).tolist(),datasets=arr.shape[0],trajectories=arr.size))
 (OUT/'trajectory_intervals.json').write_text(json.dumps(interval,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(11,4));a=out[out.section=='A'].groupby('method')[['correct_declared','wrong_declared','refused']].mean();a.plot.bar(stacked=True,ax=ax[0],rot=15);ax[0].set(ylabel='Fraction of all held-out trajectories',title='L4-A: named topology decisions');ax[0].legend(fontsize=7);a=pd.read_csv(OUT/'confusion_forced_A.csv',index_col=0).reindex(index=['G1','G2','G3','G4'],columns=['G1','G2','G3','G4'],fill_value=0);im=ax[1].imshow(a,vmin=0,vmax=1);ax[1].set(xticks=range(4),yticks=range(4),xticklabels=a.columns,yticklabels=a.index,xlabel='Forced prediction',ylabel='True structure',title='Forced classification is not identification');fig.colorbar(im,ax=ax[1]);fig.tight_layout();fig.savefig(OUT/'structure_results.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],ls=':',color='grey')
 for method in ['forced','oracle']:
  v=pd.DataFrame([z for z in calib if z['method']==method]);ax.plot(v.confidence,v.accuracy,marker='o',label=method)
 ax.set(xlim=(0,1),ylim=(0,1),xlabel='Uncalibrated score confidence',ylabel='Observed correctness',title='Reliability on untouched L4-A test answers');ax.legend();fig.tight_layout();fig.savefig(OUT/'reliability.png',dpi=170);plt.close(fig)
 # Failure examples selected deterministically: first false declaration per method.
 failures=[]
 for method in ['forced','naive_score','equivalence_guard','supplementary_audit']:
  z=out[(out.method==method)&(out.wrong_declared==1)].sort_values(['case','trajectory']).head(5);failures+=z.to_dict('records')
 (OUT/'failure_examples.json').write_text(json.dumps(failures,indent=2));print(out.groupby(['section','method']).apply(summary,include_groups=False).round(4).to_string());print('ECE',ece);print('optimizer',pd.Series([r['stop'] for r in fit]).value_counts().to_dict())
if __name__=='__main__':main()
