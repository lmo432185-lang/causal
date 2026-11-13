import json,csv
import numpy as np
from scipy.stats import norm
from config import *
from mechanisms import response,library

def main():
 bg=dict(q=np.exp(-.15*DT),c=1.5*(1-np.exp(-.15*DT))/.15,var=1.);bank=library(bg);rows=[]
 for sigma in SIGMAS:
  var=sigma*sigma*(1-bg['q']**2)/.3
  for k in KTEST:
   hh={m:response(m,[k],[10.])[0][0]*1.5 for m in LABELS}
   for ia,a in enumerate(LABELS):
    for ib,b in enumerate(LABELS):
     if ia>=ib:continue
     kl=float(np.sum((hh[a]-hh[b])**2)/(2*var));h=bank[b]['h'];dot=h@hh[a];amp=np.maximum(dot/np.sum(h*h,axis=1),0);dist=np.sum((amp[:,None]*h-hh[a])**2,axis=1);profile=float(dist.min()/(2*var))
     rows.append(dict(noise=sigma,kappa=k,pair=a+'__'+b,conditional_KL=kl,known_nuisance_optimal_accuracy=float(norm.cdf(np.sqrt(kl/2))),profile_grid_KL_A_to_B=profile))
 with (OUT/'separability.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (OUT/'equivalence_proof.md').write_text('''# Two different equivalence claims

**Representation equivalence.** For any registered source input I, first-order state recurrence is U[t+1]=r U[t]+(1-r)I[t]. Eliminating U gives the same exponential convolution and exactly the same Gaussian X distribution. Applying the same U/Y observation functions and the same do(Y=I) intervention preserves equality. No experiment can distinguish two writings of this same physical model.

**Different causal models, same passive law.** H1: Y=A step(t-t0), tau Udot=Y-U. H2: Y=A(1-exp(-(t-t0)/tau))step(t-t0), U=Y directly. Their continuous effective forces U(t) are identical, so their passive X and noisy V_U distributions are identical. With equal priors the best mechanism-label accuracy from X or (X,V_U) is 50%, irrespective of sample size. V_Y differs. Under do(Y=I), H1 filters I but H2 applies I directly, breaking passive equivalence. The intervention must act on the same physical source definition, with modularity/invariance assumed.

The four-class main library assumes a step source and known candidate equations. Its conditional model selection does not solve unknown-source causal identification. Positive fixed-nuisance KL describes distinct conditional laws; profile-grid distance is a finite-grid diagnostic, not proof of global composite identifiability. Finite-posterior abstention is not structural nonidentifiability.
''')
if __name__=='__main__':main()
