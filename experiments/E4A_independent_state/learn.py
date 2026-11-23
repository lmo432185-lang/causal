"""Only X arrays and optimizer seed accepted; no simulator or true-state imports."""
import time,numpy as np
from settings import *
from model import fit,parent_proposals
from predict import component,targets,logpdf,history_fit
from scipy.special import logsumexp
from model import PARAMS

def summarize(xv,m,ntrain):
 c=component(xv,m);s=-logpdf(c['mean'],c['cov'],targets(xv)).mean(1)/H;e=float(c['post']['ll'][:,200:].sum()-.5*PARAMS[m['kind']]*np.log(ntrain*xv.shape[1]));return s,e

def fit_direct(x,v,seed):
 models={};scores={};ev={}
 for kind in KINDS:
  m=fit(x,kind,seed+len(kind)*109);models[kind]=m;s,e=summarize(v,m,len(x));scores[kind]=s;ev[kind]=e
 return dict(models=models,selection_scores=scores,log_evidence=ev)

def fit_all(xr,xd,v,seed):
 tic=time.process_time();direct=fit_direct(xr,v,seed+10000);witness=fit_direct(xd,v,seed+20000);single=['factor1','memory2','memory3'];parent=min(single,key=lambda k:direct['selection_scores'][k].mean());poles,meta=parent_proposals(xr,direct['models'][parent],seed+30000);model2=fit(xr,'factor2',seed+40000,poles=poles);rec=dict(models={k:m for k,m in direct['models'].items()},selection_scores={k:s for k,s in direct['selection_scores'].items()},log_evidence=dict(direct['log_evidence']));rec['models']['factor2']=model2
 if model2 is None:rec['selection_scores']['factor2']=None;rec['log_evidence']['factor2']=None
 else:rec['selection_scores']['factor2'],rec['log_evidence']['factor2']=summarize(v,model2,len(xr))
 result=dict(direct=direct,witness=witness,recursive=rec,parent=parent,posterior_proposals=meta,history=history_fit(xr,v),cpu=time.process_time()-tic,training_per_route=len(xr))
 return result
