import json,hashlib
import numpy as np,pandas as pd
from config import *
from generate import draw
from mechanisms import library
from online import stream
from run import tolerance

def main():
 marker=json.loads((OUT/'TEST_START.json').read_text());unchanged={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in marker['sha256'].items()};deps=json.loads((OUT/'DEPENDENCIES_START.json').read_text());assert deps['time']<=marker['time'];unchanged.update({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in deps['sha256'].items()});assert all(unchanged.values())
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');assert len(d)==43200 and not d.duplicated(['family','arm','noise','cohort','kappa','trajectory','truth']).any();assert np.all(d.correct_commit+d.wrong_commit+d.abstain==1)
 eq=d[d.family=='passive_equivalent_pair'];assert np.all(eq[eq.arm.isin(['X','X_U'])].abstain==1)
 assert np.all(d.id_time.dropna()>=d.loc[d.id_time.notna(),'alarm'])
 ss=json.loads((OUT/'trained_settings.json').read_text());checks={}
 for ni,s in enumerate(ss):
  p=OUT/f'calibration_scores_{ni}.npz';m=OUT/f'calibration_maxima_{ni}.npz'
  with np.load(p if p.exists() else m) as z:
   for key in z.files:
    v=z[key];v=np.minimum(v[:,1:],v[:,:-1]).max(1) if v.ndim==2 else v
    from scipy.stats import binom
    th=float(np.sort(v)[int(binom.ppf(1-.05/21,len(v),.95))]);assert th==s['thresholds'][key]
  checks[str(ni)]=True
 ni=1;co=0;k=.5;s=ss[ni];data=draw(N,SIGMAS[ni],k,SEEDS['passive'][co][ni])['A_first'];bank=library(s['bg'],steps=T+1);sc,pp=stream(data['x'],bank,s['bg'],s['amp_prior'],data['U']+VSTD*data['eta'],s['sensor_variance'],'u');z=np.load(OUT/f'online_outputs_n{ni}_c{co}_k{k}.npz');key=f'four_mechanisms_n{ni}_c{co}_k{k}_truth0_X_U';se=float(np.max(abs(sc-z[key+'_scores'])));pe=float(np.max(abs(pp-z[key+'_posterior'])));assert se<1e-8 and pe<5e-8
 result=dict(frozen_files_unchanged=unchanged,row_count=len(d),row_keys_unique=True,commit_outcomes_partition=True,exact_equivalent_X_U_all_abstain=True,calibration_thresholds_reproduced=checks,replayed_score_max_error=se,replayed_posterior_float32_max_error=pe)
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
