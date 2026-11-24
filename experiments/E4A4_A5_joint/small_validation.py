"""Budget amendment: 6 fixed conditions x 3 seeds x 5 trajectories, no new fitting.

Uses existing 1000-prefix models and the first five of the original paired test-B
trajectories. This is exploratory validation, not the cancelled full short test B.
"""
from pathlib import Path
import sys,json,gzip,hashlib,time,numpy as np,pandas as pd
BASE=Path(__file__).parent;SHORT=BASE/'short';sys.path.insert(0,str(SHORT))
import evaluate as ev
from producer import SYSTEMS,generate as original_generate,digest
OUT=BASE/'small_validation';OUT.mkdir(exist_ok=True)
CONDS=['A0','B1','C0','D0','E1','CE'];SEEDS=[4011,4012,4013]
def limited_generate(index,seed,stage,count):return original_generate(index,seed,stage,20)[:5].copy()
def main():
 assert (BASE/'USER_SMALL_SCALE_AMENDMENT.json').exists()
 jobs=[j for j in json.loads((SHORT/'jobs.json').read_text()) if SYSTEMS[j['index']]['id'] in CONDS and j['seed'] in SEEDS]
 (OUT/'PLAN.json').write_text(json.dumps(dict(time=time.time(),conditions=CONDS,seeds=SEEDS,trajectories_per_case=5,existing_test_length=5000,model_training_length=1000,only_reuse_frozen_models=True,prospective_choice_not_based_on_gains=True,exploratory=True,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 ev.NTEST=5;ev.generate=limited_generate;rows=[];controls=[];hashes={}
 for j in jobs:
  a,c,h=ev.work(j,'B');rows+=a;controls+=c;hashes[j['case']]=h
 d=pd.DataFrame(rows);d.to_csv(OUT/'test_B_small.csv.gz',index=False);pd.DataFrame(controls).to_csv(OUT/'controls_B_small.csv.gz',index=False)
 long=pd.read_csv(BASE/'results/test_B.csv.gz');paired=d.merge(long,on=['case','seed','method','trajectory'],suffixes=('_1000','_5000'),validate='one_to_one');paired['length_gain']=pd.to_numeric(paired.NLL_1000)-paired.NLL_5000;paired.to_csv(OUT/'paired_length.csv.gz',index=False)
 p=d.pivot(index=['case','seed','trajectory'],columns='method',values='NLL').astype(float);p['dual_gain']=p.BestSingle-p.E;p['full_reference_gain']=p.StrongestSingle-p.E;p['condition']=p.index.get_level_values('case').str.split('_').str[0];p.groupby('condition')[['dual_gain','full_reference_gain']].agg(['mean','min','max']).to_csv(OUT/'descriptive_gains.csv')
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='COMPLETE.json'};(OUT/'COMPLETE.json').write_text(json.dumps(dict(time=time.time(),rows=len(d),models=len(jobs),no_new_fitting=True,independent_train_seeds=3,exploratory_no_significance_claim=True,hashes=hashes,files=files),indent=2));print('Small validation complete',len(d),flush=True)
if __name__=='__main__':main()
