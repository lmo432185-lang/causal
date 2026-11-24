import numpy as np,time,json
from bayes import likelihood,to_theta
from settings import ROOT
import native
r=np.random.default_rng(991919);x=np.ascontiguousarray(r.normal(size=(20,5000)));F=np.diag([.87,.45]);k=np.array([.6,.4]);m=dict(F=F,Q=np.diag(k*(1-np.diag(F)**2)),C=np.ones(2),R=.4,P=np.diag(k),a=0.,scale=1.);z=to_theta(m);a=likelihood(x,z);b=native.run(x,m)['ll'].sum();assert abs(a-b)<1e-6;tic=time.process_time()
for i in range(100):likelihood(x,z)
(ROOT/'BAYES_DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),unrelated_data=True,exact_likelihood_error=float(abs(a-b)),cpu_per_call=(time.process_time()-tic)/100,passed=True),indent=2));print('Exact Gaussian likelihood validated',abs(a-b))
