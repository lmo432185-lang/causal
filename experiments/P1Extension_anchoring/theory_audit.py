"""Pretraining finite equivalence audit. It is not a proof that no other alternatives exist."""
import json,time,itertools
import numpy as np
from settings import ROOT,DELAYS,INPUT_STD
from model import plant,covariance
from private_generator import CELLS,theta

def spectrum(th):
 a,b,c,u,v,q,R=th[:7];w=2*np.pi*np.arange(1024)/1024;D=[1+r*r-2*r*np.cos(w) for r in [a,b,c]];return R+(q+v*v/D[1]+u*u*v*v/(D[0]*D[1]))/D[2]
def aliases(th):
 a,b,c,u,v,q,R=th[:7];co=np.cos(np.linspace(0,np.pi,65));D=lambda r:1+r*r-2*r*co;N=q*D(a)*D(b)+v*v*D(a)+u*u*v*v;B=np.stack([np.ones(len(co)),co,co**2],1);nc=np.linalg.lstsq(B,N,rcond=None)[0];out=[]
 for aa,bb,cc in itertools.permutations([a,b,c]):
  pc=np.linalg.lstsq(B,D(aa)*D(bb),rcond=None)[0];qq=nc[2]/pc[2];F=nc-qq*pc;vv2=-F[1]/(2*aa);uu2=F[0]/vv2-1-aa*aa if vv2>0 else -1
  if qq>0 and vv2>0 and uu2>1e-9:
   alt=np.array([aa,bb,cc,np.sqrt(uu2),np.sqrt(vv2),qq,R,1.,1.,1.]);assert np.max(abs(spectrum(alt)-spectrum(th)))/np.max(spectrum(th))<1e-9;out.append(alt)
 return out

def main():
 rows=[]
 for cell in CELLS:
  for policy in ['passive','active']:
   true=theta(cell['params']);d=cell['params']['delay'];marg=true.copy()
   if policy=='active':marg[3]*=np.sqrt(1+INPUT_STD**2);marg[7]=np.sqrt(1+INPUT_STD**2)
   for kind in ['E0','E1','E2','E3','E4Y','E4Z','E4YZ']:
    target=true.copy() if kind=='E3' else marg.copy()
    if kind=='E3' and policy=='passive':
     # Known all-zero input does not identify source scale; audit as ordinary X-only candidates.
     auditkind='E0'
    else:auditkind=kind
    C0=covariance(target,d,auditkind,False);kept=[]
    for alt in aliases(target):
     if kind=='E1' and not (alt[0]>alt[1]>alt[2]):continue
     if kind=='E2' and abs(alt[2]-true[2])>1e-8:continue
     for lag in DELAYS:
      a=alt.copy()
      if auditkind=='E3':a[9]=target[9]*target[3]*target[4]/(a[3]*a[4])
      if auditkind.startswith('E4'):
       pa=plant(a,lag,'E0')[3];pt=plant(target,d,'E0')[3]
       if auditkind in ['E4Y','E4YZ']:a[7]=target[7]*np.sqrt(pt[0,0]/pa[0,0])
       if auditkind in ['E4Z','E4YZ']:a[8]=target[8]*np.sqrt(pt[1,1]/pa[1,1])
      # X-only contrasts do not use auxiliary h loadings, so target h is irrelevant.
      C=covariance(a,lag,auditkind,False);err=float(np.max(abs(C-C0))/max(1,np.max(abs(C0))))
      if err<1e-8:kept.append(dict(theta=a.tolist(),delay=lag,relative_covariance_error=err))
    Cfull,Dfull=covariance(target,d,auditkind,True);free=[0,1,2,3,4,5,6]
    if kind=='E2':free.remove(2)
    if auditkind=='E3':free.append(9)
    if auditkind in ['E4Y','E4YZ']:free.append(7)
    if auditkind in ['E4Z','E4YZ']:free.append(8)
    L=np.linalg.cholesky(Cfull);wh=[]
    for k in free:wh.append(np.linalg.solve(L,np.linalg.solve(L,Dfull[k]).T).T*max(abs(target[k]),.05))
    wh=np.array(wh).reshape(len(free),-1).T;sv=np.linalg.svd(wh,compute_uv=False);rank=int(np.sum(sv>sv[0]*1e-9))
    rows.append(dict(local_covariance_jacobian_rank=rank,local_free_parameters=len(free),local_smallest_singular_value=float(sv[-1]),tag=cell['tag'],policy=policy,method=kind,valid_order=cell['valid_order'],retained_known_equivalents=len(kept),retained_distinct_pole_orders=len({tuple(np.round(v['theta'][:3],8)) for v in kept}),retained_delays=sorted({v['delay'] for v in kept}),candidates=kept,scope='finite exact pole-permutation and delay family; uniqueness beyond this family not proved'))
 result=dict(time=time.time(),before_data_generation=False,before_affected_sparse_training=True,unaffected_Xonly_models_already_trained=True,records=rows,independent_edge_delay_warning='E3 can at best identify dYZ+dZX when delays need not equal; common-delay assumption is retained only to compare with P0-P1',scale_warning='unit innovation coordinates alone do not anchor physical hidden units; auxiliary observation loadings and known input lambda remain explicit',not_global_identifiability_proof=True);(ROOT/'PRETRAIN_THEORY_AUDIT.json').write_text(json.dumps(result,indent=2))
 for row in rows:
  if row['tag']=='baseline':print(row['policy'],row['method'],row['retained_known_equivalents'],row['retained_delays'])
if __name__=='__main__':main()
