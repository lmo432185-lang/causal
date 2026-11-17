"""Independent follow-up positive control designed from algebra, not test answers."""
import json,time,hashlib
import numpy as np
from config import R,D,BASE,SEEDS
from generate import generate_one

def check_separation():
 w=np.linspace(0,np.pi,101);z=np.exp(-1j*w);poles=[.8,.7,.5];ds=[abs(1-a*z)**2 for a in poles];den=np.prod(ds,axis=0);B=np.stack([np.cos(k*w) for k in range(4)],1);a,c,b=poles;u=.2;v=.4;alpha=2.5;F=abs(u*(1-c*z)+v*alpha*z)**2+v*v*abs(1-a*z)**2;S=(.25+F/(ds[0]*ds[1]))/ds[2]+.0625;coef=np.linalg.lstsq(B,S*den,rcond=None)[0];dc=np.linalg.lstsq(B,den,rcond=None)[0];r=coef[3]/dc[3];rem=coef-r*dc;checks=[]
 for t in range(3):
  h=[i for i in range(3) if i!=t];d=np.linalg.lstsq(B,ds[h[0]]*ds[h[1]],rcond=None)[0];qx=rem[2]/d[2];force=rem-qx*d;M=force[0];L=-force[1]/2
  for s in h:
   g2=L/poles[s];al2=M/g2-1-poles[s]**2 if g2>0 else None;valid=bool(qx>=0 and g2>0 and al2 is not None and al2>=0);checks.append(dict(target_pole=poles[t],source_pole=poles[s],implied_qx=float(qx),implied_output_gain_squared=float(g2),implied_internal_gain_squared=float(al2) if al2 is not None else None,legal_chain_representation=valid))
 assert not any(v['legal_chain_representation'] for v in checks);return checks
if __name__=='__main__':
 checks=check_separation();protocol=dict(time=time.time(),scope='independent positive follow-up, not modification of original99 conditions',N=1000,split=[600,200,200],T=1000,seeds=SEEDS,truth='G3',change='alpha2.5 vs baseline.5',reason='algebraically distinct from positive-AR parallel/cascade candidates after all target-pole assignments; original test truth not used to tune algorithm',fit='unchanged4starts/MAXITER40 andall original scores/thresholds',confidence_scope='can distinguish mixed family only conditional on specified candidate restrictions, not named hidden identity in unrestricted world',proof=checks);(R/'POSITIVE_PROTOCOL.json').write_text(json.dumps(protocol,indent=2));files=['positive_prepare.py','positive_fit.py','positive_evaluate.py','POSITIVE_PROTOCOL.json'];(R/'POSITIVE_REGISTERED.json').write_text(json.dumps(dict(time=time.time(),files={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in files}),indent=2));public=[];private=[];cell=dict(id='case100',section='F_positive',truth='G3',kind='G3',tag='alpha2.5_positive',params={**BASE,'alpha':2.5})
 for seed in SEEDS:
  case=f'case100_{seed}';dest=D/case;dest.mkdir(exist_ok=True);x,s,m=generate_one(cell,seed);np.save(dest/'X.npy',x);np.savez_compressed(dest/'sealed_states.npz',physical=s);public.append(dict(case=case,seed=seed,X_path=str((dest/'X.npy').relative_to(R))));private.append(dict(**cell,case=case,seed=seed,generation=m))
 (R/'positive_public_jobs.json').write_text(json.dumps(public,indent=2));(D/'positive_sealed_manifest.json').write_text(json.dumps(private,indent=2));print('Independent positive control registered and generated',flush=True)
