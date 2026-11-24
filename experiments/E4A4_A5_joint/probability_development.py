import json,time,numpy as np
from scipy.stats import norm
from scipy.linalg import solve_discrete_lyapunov
from probability import marginal,physical_scores,correlated_alias
from predict import component
from settings import ROOT

def main():
 rng=np.random.default_rng(88891);mu=np.zeros((5000,1));var=np.ones_like(mu);w=np.ones_like(mu);y=rng.normal(size=5000);r,_,_=marginal(mu,var,w,y);assert abs(np.mean(r['width90'])-2*norm.ppf(.95))<1e-6
 P=12000;means=np.broadcast_to(np.array([[-.7,.3],[.6,-.4]]),(P,2,2));cov=np.broadcast_to(np.array([[[1.,.3],[.3,.5]],[[.6,-.2],[-.2,.8]]]),(P,2,2,2));weights=np.broadcast_to(np.array([.4,.6]),(P,2));choice=rng.choice(2,P,p=[.4,.6]);draw=np.array([rng.multivariate_normal(means[i,k],cov[i,k]) for i,k in enumerate(choice)]);q=physical_scores(means,cov,weights,draw);cover=float(q['joint_cover90'].mean());assert abs(cover-.9)<.015
 F=np.diag([.87,.39]);Q=np.diag([.2,.4]);m=dict(F=F,Q=Q,C=np.ones(2),R=.6,P=solve_discrete_lyapunov(F,Q),scale=1.,a=0.,kind='factor2');x=rng.normal(size=(2,500));c0=component(x,m);errs=[]
 for rho in [-.99,-.5,0,.5,.99]:
  c=component(x,correlated_alias(m,rho));err=max(np.max(abs(c['mean']-c0['mean'])),np.max(abs(c['cov']-c0['cov'])),np.max(abs(c['post']['ll']-c0['post']['ll'])));assert err<1e-8;errs.append(float(err))
 (ROOT/'PROBABILITY_DEVELOPMENT.json').write_text(json.dumps(dict(time=time.time(),unrelated_synthetic_data=True,all_passed=True,mixture_joint90_empirical=cover,null_likelihood_max_error=max(errs),normal_interval_error=float(abs(r['width90'].mean()-2*norm.ppf(.95)))),indent=2));print('Probability development passed',cover)
if __name__=='__main__':main()
