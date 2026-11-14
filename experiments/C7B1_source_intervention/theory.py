import csv,json
import numpy as np
from scipy.stats import norm
from config import *
from physics import *

def main():
 commands=make_commands(100,420000);rows=[]
 for ni,sigma in enumerate(SIGMAS):
  var=sigma*sigma*(1-np.exp(-.3*DT))/.3
  for k in KTEST:
   for key in INPUTS[1:]:
    I=commands[key];ha,ua=response(I,np.full(100,k));hb,ub=response(I,np.ones(100),direct=True);d=np.cumsum((ha-hb)**2,axis=1)/(2*var);du=np.cumsum((ua-ub)**2,axis=1)/(2*VSTD**2)
    for end in [12.,14.,25.]:
     j=int(end/DT)-1;kl=d[:,j];rows.append(dict(noise=sigma,tau=k,input=key,time=end,mean_KL_X=float(kl.mean()),mean_known_tau_optimal_accuracy_X=float(norm.cdf(np.sqrt(kl/2)).mean()),mean_KL_X_U=float((kl+du[:,j]).mean())))
 with (OUT/'conditional_information.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 # Finite-data X distinguishability vanishes as the allowed lag tends to zero.
 I=commands['I2_pulse'];hd,_=response(I,np.ones(100),direct=True);lim=[]
 for k in [.1,.01,.001,.0001]:
  hf,_=response(I,np.full(100,k));lim.append(dict(tau=k,mean_squared_force_integral_distance=float(np.mean(np.sum((hf-hd)**2,axis=1)))))
 (OUT/'small_lag_limit.json').write_text(json.dumps(lim,indent=2))
if __name__=='__main__':main()
