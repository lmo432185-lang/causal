import numpy as np,json,time
from scipy.special import logsumexp
from forecast_joint import region
from predict import logpdf
from settings import ROOT
rng=np.random.default_rng(938173);N=6000;mu=np.array([[0.,1.,-.2],[1.,0.,.4]]);cov=np.array([[[1.,.3,.1],[.3,1.,.4],[.1,.4,1.]],[[.8,-.1,.3],[-.1,.9,.2],[.3,.2,1.2]]]);w=np.array([.4,.6]);which=rng.choice(2,N,p=w);y=np.array([rng.multivariate_normal(mu[k],cov[k]) for k in which])[:,None];r=dict(mean=np.broadcast_to(mu,(N,1,2,3)),cov=cov[:,None],origin_weights=np.broadcast_to(w,(N,1,2)));out=region(r,y);ll=np.stack([logpdf(r['mean'][:,:,k],r['cov'][k],y) for k in range(2)],2);expected=-logsumexp(np.log(w)+ll,2).mean(1)/3;err=float(abs(expected-out['joint_NLL']).max());assert err<1e-10 and abs(out['joint_cover90'].mean()-.9)<.02;(ROOT/'FORECAST_JOINT_DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),no_formal_data=True,maximum_density_error=err,coverage90=float(out['joint_cover90'].mean()),passed=True),indent=2));print('Joint forecast development passed',err)
