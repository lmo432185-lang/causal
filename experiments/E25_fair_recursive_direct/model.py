"""X-only linear representations. No generator/system/latent-state access."""
import time,json
import numpy as np
from scipy.fft import dct
from settings import *
from core import *

FIXED=dct(np.eye(41),type=2,norm='ortho',axis=0).T[:,np.arange(0,41,2)]
RANDOM=np.linalg.qr(np.random.default_rng(250029).normal(size=(41,21)))[0]

def raw(x):
 c,p,y=base_arrays(x,FULL);return c,p,y

def recparts(x,short):
 w=wseries(x,short);wc,wp,wf=recursive_arrays(w);return wc,wp,wf

def concat(*args):return np.concatenate(args,axis=-1)

def encode_direct(enc,c,p,mode):
 past=p if mode=='full' else concat(c,p)@(FIXED if mode=='fixed' else RANDOM)
 return apply(enc,c,past)

def features(x,m):
 c,p,y=raw(x);wc,wp,wf=recparts(x,m['short']);a=apply(m['rec'],wc,wp)[:,:,m['candidate']:m['candidate']+1]
 f=dict(current=c,W1=concat(c,wc),recursive=concat(c,wc,a),full_W1=concat(c,wc,wp),full_X=concat(c,p))
 for mode in ['full','fixed','random']:
  z=encode_direct(m['direct'][mode],c,p,mode);f['direct_'+mode]=concat(c,z[:,:,:2]);f['first_'+mode]=concat(c,z[:,:,:1])
 return f,y

def train(x,v):
 cpu0=time.process_time();wall0=time.perf_counter();c,p,y=raw(x);vc,vp,vy=raw(v);budget={};timing={}
 t=time.process_time();w=time.perf_counter();sc,sp,sy=base_arrays(x,P);short=learn(sc,sp,sy,k=1);wc,wp,wf=recparts(x,short);vwc,vwp,vwf=recparts(v,short);rec=learn(wc,wp,wf);a=apply(rec,wc,wp);va=apply(rec,vwc,vwp);candidate_heads=[];scores=[]
 for j in range(KMAX):
  h=fit_head(concat(c,wc,a[:,:,j:j+1]),y);candidate_heads.append(h);scores.append(float(nll_series(h,concat(vc,vwc,va[:,:,j:j+1]),vy).mean()))
 j=int(np.argmin(scores));heads={'recursive':candidate_heads[j]};timing['recursive']=dict(cpu=time.process_time()-t,wall=time.perf_counter()-w,encoder_fits=2,candidate_head_fits=6,nominal_filter_coefficients=42,head_parameters=95)
 direct={}
 for mode in ['full','fixed','random']:
  t=time.process_time();w=time.perf_counter();past=p if mode=='full' else concat(c,p)@(FIXED if mode=='fixed' else RANDOM);enc=learn(c,past,y,k=2);z=apply(enc,c,past);h=fit_head(concat(c,z[:,:,:2]),y);direct[mode]=enc;heads['direct_'+mode]=h;heads['first_'+mode]=fit_head(concat(c,z[:,:,:1]),y);timing['direct_'+mode]=dict(cpu=time.process_time()-t,wall=time.perf_counter()-w,encoder_fits=1,candidate_head_fits=1,nominal_filter_coefficients=82 if mode=='full' else 44,head_parameters=95)
 heads['current']=fit_head(c,y);heads['W1']=fit_head(concat(c,wc),y);heads['full_W1']=fit_head(concat(c,wc,wp),y);heads['full_X']=fit_head(concat(c,p),y)
 return dict(short=short,rec=rec,candidate=j,candidate_selection_NLL=scores,direct=direct,heads=heads,timing=timing,total_cpu=time.process_time()-cpu0,total_wall=time.perf_counter()-wall0,training_trajectories=len(x))

def save(p,m):p.write_text(json.dumps(pack(m)))
def load(p):return unpack(json.loads(p.read_text()))
