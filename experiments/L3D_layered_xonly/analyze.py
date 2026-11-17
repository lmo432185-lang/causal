import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;O=R/'results';e=pd.DataFrame(json.loads((O/'edge_summary.json').read_text()));m=pd.DataFrame(json.loads((O/'models.json').read_text()));d=pd.read_csv(O/'state_metrics.csv.gz');e.to_csv(O/'edge_summary.csv',index=False);d.groupby(['group','state','method'])[['MSE','NMSE','corr']].mean().to_csv(O/'state_summary.csv');d.groupby(['group','seed','state','method'])[['MSE','NMSE','corr']].mean().to_csv(O/'state_by_seed.csv');m.to_csv(O/'model_summary.csv',index=False)
# Exact oracle graph check: labels used only in evaluation.
expected={'D1_direct':{'Y->X'},'D2_indirect':{'Y->Z','Z->X'},'D3_mixed':{'Y->Z','Z->X','Y->X'}};check=[]
for (group,seed),g in e[(e.family=='oracle_true_states')&(e.control=='all_node_histories')&(e.lag==1)].groupby(['group','seed']):
 links={f'{v.source}->{v.target}' for _,v in g[g.test_supported].iterrows()};tp=len(links&expected[group]);fp=len(links-expected[group]);fn=len(expected[group]-links);check.append(dict(group=group,seed=int(seed),links=sorted(links),exact_graph_match=bool(links==expected[group]),TP=tp,FP=fp,FN=fn))
(O/'oracle_graph_evaluation.json').write_text(json.dumps(check,indent=2))
r=json.loads((O/'rotations.json').read_text());rotation=[]
for group in m.group.unique():
 for seed in [101,202,303]:
  a=[q for q in r if q['group']==group and q['seed']==seed];base=set(a[0]['supported_edges']);rotation.append(dict(group=group,seed=seed,distinct_graphs=len({tuple(sorted(v['supported_edges'])) for v in a}),max_prediction_difference=max(v['forecast_max_difference'] for v in a),graphs=[dict(angle=v['angle'],edges=v['supported_edges']) for v in a]))
(O/'rotation_summary.json').write_text(json.dumps(rotation,indent=2))
# Comparisons of h1 direct candidates before/after full X history control.
primary=e[(e.family=='selected')&(e.stage=='direct')&(e.horizon==1)];primary.groupby(['group','control'])[['validation_gain','test_gain','test_supported']].mean().to_csv(O/'direct_gain_summary.csv');direc=e[(e.family=='selected')&(e.stage=='direction')];direc.groupby(['group','control','lag'])[['validation_gain','test_gain','test_supported']].mean().to_csv(O/'direction_gain_summary.csv')
fig,ax=plt.subplots(1,2,figsize=(11,4));s=primary.groupby(['group','control']).test_gain.mean().unstack('control');s.plot.bar(ax=ax[0],rot=10);ax[0].set(ylabel='Mean held-out relative MSE reduction',title='Conditional latent -> next X');ax[0].axhline(.01,color='grey',ls=':');ax[0].legend(fontsize=8);s=d.groupby(['group','state','method']).NMSE.mean().unstack('method').reindex(columns=['current_X','L_only','joint_X_L','history20']);s.plot.bar(ax=ax[1],rot=25);ax[1].set(ylabel='Normalized hidden reconstruction MSE',title='Reconstruction after representation freeze');ax[1].legend(fontsize=6);fig.tight_layout();fig.savefig(O/'conditional_and_recovery.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));ranks=m.set_index(['group','seed'])['rank'];ranks.plot.bar(ax=ax);ax.set(ylabel='Selected extra predictive dimensions',title='Validation X chooses rank; count is not physical hidden count');fig.tight_layout();fig.savefig(O/'ranks.png',dpi=170);plt.close(fig)
# Oracle horizon contrast: indirect upstream Y affects farther future even without direct path.
v=e[(e.family=='oracle_hidden')&(e.control=='full_X20')];v.groupby(['group','source','horizon'])[['validation_gain','test_gain','test_supported']].mean().to_csv(O/'oracle_horizon_contrast.csv')
print(m[['group','seed','rank','validated_direct_states']].to_string(index=False));print('direct',primary.groupby(['group','control'])[['test_gain','test_supported']].mean().to_string());print('direction',direc.groupby(['group','control','lag'])[['test_gain','test_supported']].mean().to_string());print('oracle',json.dumps(check));print('rotation',[(v['group'],v['seed'],v['distinct_graphs']) for v in rotation]);print(d.groupby(['group','state','method'])[['NMSE','corr']].mean().round(4).to_string());print('oracle horizons',v.groupby(['group','source','horizon']).test_gain.mean().round(4).to_string())
# Avoid counting a mere swap of latent labels as a new topology.
def canonical_links(links):
 def swap(edge):
  a,b=edge.split('->');f=lambda x:'L2' if x=='L1' else 'L1' if x=='L2' else x
  return f(a)+'->'+f(b)
 return min(tuple(sorted(links)),tuple(sorted(swap(v) for v in links)))
for v in rotation:v['distinct_graphs_up_to_latent_swap']=len({canonical_links(g['edges']) for g in v['graphs']})
(O/'rotation_summary.json').write_text(json.dumps(rotation,indent=2))
