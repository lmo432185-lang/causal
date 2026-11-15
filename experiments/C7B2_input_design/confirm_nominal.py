"""Post-test reliability follow-up; frozen thresholds/policies; no reselection."""
import json
import numpy as np,pandas as pd
from config import *
from physics import *
from inference import stream
from run import seed_for

def main():
 sel=json.loads((OUT/'SELECTION.json').read_text());settings=json.loads((OUT/'trained_settings.json').read_text());names=sorted({REFERENCE}|{x for x in sel['policies'].values() if x is not None});rows=[]
 for name in names:
  program=PROGRAMS[IDS.index(name)]
  for ni,sigma in enumerate(SIGMAS):
   flags=[]
   for batch in range(50):
    seed=seed_for(630000+100*ni,name,batch);rng=np.random.default_rng(np.random.SeedSequence([seed,19]));truth=rng.integers(0,2,100);tau=rng.choice(KTEST,100);I=command(100,program,seed);d=simulate(I,sigma,tau,truth,seed);_,z=stream(d['x'],I,settings[ni]['bg']);v=np.minimum(z[:,1:],z[:,:-1]).max(1);flags.extend(v>settings[ni]['thresholds'][name])
   n=len(flags);count=int(np.sum(flags));p=count/n;zz=1.96;den=1+zz*zz/n;center=(p+zz*zz/(2*n))/den;half=zz*np.sqrt(p*(1-p)/n+zz*zz/(4*n*n))/den;rows.append(dict(program=name,noise=sigma,n=n,count=count,fpr=p,ci95_low=center-half,ci95_high=center+half));print(name,sigma,p,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'post_test_nominal_confirmation.csv',index=False)
 (OUT/'post_test_nominal_confirmation_design.json').write_text(json.dumps(dict(post_test=True,reason='Selected E4 square has5.44% nominal anomaly point in dependent main test; follow up independently without retuning',n_per_program_noise=5000,seed_base=630000,seed_derivation='run.seed_for(base+100*noise_index,program,batch)',no_threshold_change=True,no_policy_change=True,not_part_of_primary_inferential_family=True),indent=2))
if __name__=='__main__':main()
