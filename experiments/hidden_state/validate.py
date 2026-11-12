import json,numpy as np
from run import information,OUT
# Exact Gaussian conditioning reference: z0~N(0,1), zt=.8 z(t-1)+w,
# increments X(t+1)-X(t)=1.2 zt+v.
rng=np.random.default_rng(87); T=80; x=np.cumsum(rng.normal(size=(2,T)),axis=1)
a=np.array([0,0,0,.8,0]); b=np.array([0,0,0,1.2,0]); q=.2; r=.5
aci,cir=information((a,b,q,r),x)
var=np.empty(T); var[0]=1
for t in range(1,T): var[t]=.8**2*var[t-1]+q
cov=np.array([[var[min(i,j)]*.8**abs(i-j) for j in range(T)] for i in range(T)])
C=1.2**2*cov[:-1,:-1]+np.eye(T-1)*r
cross=1.2*cov[0,:-1]; sp=1-cross@np.linalg.solve(C,cross)
for j in range(2):
 sm=cross@np.linalg.solve(C,np.diff(x[j])); exact=.5*(-np.log(sp)+sp+sm**2-1)
 np.testing.assert_allclose(aci[j,0],exact,rtol=1e-6,atol=1e-8)
zero,_=information((a,np.zeros(5),q,r),x)
np.testing.assert_allclose(zero,0,atol=1e-10)
assert np.isfinite(aci).all() and (aci>=-1e-8).all()
(OUT/'validation.json').write_text(json.dumps({'exact_gaussian_conditioning':'passed','zero_latent_observation_ACI':'passed','finite_nonnegative_ACI':'passed'},indent=2))
print('All Gaussian inference checks passed')
