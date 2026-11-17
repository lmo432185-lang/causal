from pathlib import Path
import numpy as np
R=Path(__file__).parent;O=R/'results';O.mkdir(exist_ok=True);D=R/'data';D.mkdir(exist_ok=True)
SEEDS=[101,202,303];N=1000;T=1000;TRAIN=600;VALID=200;TEST=200;INITS=4;MAXITER=40;SCORE_START=100;CONF=.95
BASE=dict(a=.8,c=.7,b=.5,alpha=.5,g1=.4,g2=.8,qy=.25,qz=.25,qx=.25,delay=1,obs=.25)
CELLS=[]
def add(section,truth,change=None,tag='base',explicit=None):
 p={**BASE,**(change or {})};kind=truth
 if explicit is not None:p=explicit.copy()
 elif kind=='G1':p.update(alpha=0.,g1=.6,g2=.6)
 elif kind=='G2':p.update(g1=0.)
 elif kind=='G4':p.update(g1=.8,g2=0.)
 CELLS.append(dict(id=f'case{len(CELLS):03}',section=section,truth=truth,kind=kind,tag=tag,params=p))
for g in ['G1','G2','G3','G4']:add('A',g)
for noise in [.05,.75]:
 for g in ['G1','G2','G3','G4']:add('B',g,dict(obs=noise),f'obs{noise}')
for tag,a,c in [('separated',.9,.3),('similar',.75,.73)]:
 for g in ['G1','G2','G3','G4']:add('C',g,dict(a=a,c=c),tag)
for alpha in [.2,.8]:
 for g in ['G2','G3','G4']:add('D',g,dict(alpha=alpha),f'alpha{alpha}')
for g in ['G2','G3','G4']:add('D',g,dict(delay=2),'delay2')
# Exact permutation pair, matched by named Y/Z exchange.
add('E','G2',tag='named_direction_swap',explicit={**BASE,'g1':0.})
add('E','G4',tag='named_direction_swap',explicit={**BASE,'a':.7,'c':.8,'g1':.8,'g2':0.})
# Parallel independent sources and noisy cascade have equal stationary X spectra.
a,c=.8,.7;alpha=.2;mixA=alpha**2*.25/((1+c*c)-(c/a)*(1+a*a));mixB=.25-(c/a)*mixA
add('E','G1',tag='parallel_chain_equal',explicit={**BASE,'alpha':0.,'g1':float(.8*np.sqrt(mixA/.25)),'g2':float(.8*np.sqrt(mixB/.25))})
add('E','G2',tag='parallel_chain_equal',explicit={**BASE,'alpha':alpha,'g1':0.})
PROTOCOL=dict(name='L4 new definition: two named candidate hidden variables, unknown-parameter X-only structural SSM',seeds=SEEDS,cells=CELLS,n=N,length=T,split=[600,200,200],stationary_initial=True,variance_matching='each physical X normalized by its theoretical stationary std at generation, X variance1; normalization not supplied to fitter; measurement noise independent',learner='observations only, no true YZ/graph/parameters; unit latent innovation covariance is a coordinate/scale gauge, not truth values',fit='stationary constrained generalized EM, closed-form/box-constrained transition blocks; stationary P0 refreshed; observed-likelihood rollback safeguard; warmup covariance converged1e-12 then fast stable recurrences',topologies=['G1 parallel','G2 Y->Z->X','G3 mixed','G4 Z->Y->X'],initializations=INITS,max_iterations=MAXITER,convergence='per-observation likelihood change<3e-5 after8iterations',delay_search='G3 2initializations each d1,d2; G1/G2/single4 each; pure-cascade delay has same stationary X law, fit d1 canonical; G4 exact mirrored four G2 EM trajectories, separately checked',bounds=dict(AR=[0,.98],coupling=[0,3],qx_R=[1e-5,5]),selection='per-candidate choose initialization/delay by validation LL-0.5*k*log(nval); no labels; all hyperparameters fixed',confidence='per-test trajectory softmax(conditional loglik-0.5*k*log(900)); uncalibrated score, not Bayesian true-graph probability',rejection='score>=.95 plus validation compatibility and estimated-law equivalence closure audit; required fitted path coefficient<1e-3 is boundary-ambiguous; separately retain naive-score decision',equivalence_audit='known permutation of hidden names; fitted stationary spectral identities mapping parallel/chain/mixed into other allowed candidate topologies; no truth used',baseline='single hidden state, same4 initializations',oracle='only after global X-only decisions frozen; Y,Z observed; separate structural reference',alignment='only after all X-only predictions and thresholds frozen, fit joint linear map on validation trueYZ then evaluate test; alignment never changes graph scores',statistics='independent trajectories and3 data seeds; no timepoint IID; report optimizer caps and failures, no calibrating temperature from labels',not_claimed=['arbitrary hidden causal discovery','globally converged MLE for every start','calibrated graph posterior','dynamic failure warning'])
