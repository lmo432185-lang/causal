import json,hashlib,inspect
import numpy as np,pandas as pd
from config import *
from physics import *
from inference import stream,commit
from run import threshold

def main():
 mark=json.loads((OUT/'TEST_START.json').read_text());same={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in mark['sha256'].items()};assert all(same.values())
 d=pd.read_csv(OUT/'trajectory_metrics.csv.gz');assert len(d)==38400;assert not d.duplicated(['input','arm','noise','cohort','kappa','trajectory','truth']).any();assert np.all(d.correct_first+d.wrong_first+d.refuse==1)
 passive=d[d.input=='I0_passive'];assert np.all(passive.refuse==1);assert passive.final_hard_accuracy.mean()==.5;assert np.all(d.first_id_time.dropna()>=START)
 assert np.allclose(d[d.input!='I0_passive'].energy_total,ENERGY)
 settings=json.loads((OUT/'trained_settings.json').read_text());reconstructed={}
 for ni,s in enumerate(settings):
  p=OUT/f'calibration_scores_{ni}.npz';mx=OUT/f'calibration_maxima_{ni}.npz'
  with np.load(p if p.exists() else mx) as z:
   for arm in ARMS:
    v=z[arm];v=np.minimum(v[:,1:],v[:,:-1]).max(1) if v.ndim==2 else v
    from scipy.stats import binom
    t=float(np.sort(v)[int(binom.ppf(1-.05/6,len(v),.95))]);assert t==s['thresholds'][arm]
  reconstructed[str(ni)]=True
 ni=1;co=1;k=.5;seed=SEEDS['test'][co][ni];I=make_commands(N,seed)['I3_random'];s=settings[ni];data=simulate(I,SIGMAS[ni],np.full(N,k),0,seed);p,z=stream(data['x'],I,s['bg'],data['U']+VSTD*data['eta'],s['sensor_variance']);saved=np.load(OUT/f'online_outputs_n{ni}_c{co}_k{k}.npz');delta=float(np.max(abs(p-saved['I3_random_truth0_X_U_posterior'])));se=float(np.max(abs(z-saved['I3_random_truth0_X_U_nominal_score'])));assert delta<5e-8 and se<1e-8
 # Actual source wave is same for both mechanisms; commanded X changes follow unchanged observation equation.
 a=simulate(I,.35,np.full(N,k),0,seed);b=simulate(I,.35,np.full(N,k),1,seed);assert np.array_equal(a['command'],b['command']) and np.array_equal(a['Y'],b['Y']);assert not np.array_equal(a['x'],b['x'])
 q=np.exp(-.15*DT);sd=.35*np.sqrt((1-q*q)/.3);h,_=response(I,np.full(N,k));noise=innovations(N,.35,seed)[0];res=a['x'][:,1:]-q*a['x'][:,:-1]-1.5*(1-q)/.15+h;assert np.max(abs(res-sd*noise))<1e-12
 report=dict(frozen_files_unchanged=same,records=len(d),commit_outcome_partition=True,passive_all_refuse_and_forced_chance=True,equal_active_energy=True,calibration_thresholds_reproduced=reconstructed,replayed_probability_float32_max_error=delta,replayed_nominal_score_max_error=se,actual_source_command_identical_across_mechanisms=True,observation_equation_unchanged=True,inference_interface=str(inspect.signature(stream)))
 (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
