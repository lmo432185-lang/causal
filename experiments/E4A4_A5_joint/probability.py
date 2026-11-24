"""Exact finite Gaussian-mixture scores and Rosenblatt joint credible regions."""
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.stats import norm,chi2
from scipy.linalg import solve_discrete_lyapunov
LEVELS=[.5,.8,.9,.95];TIMES=np.arange(250,5000,500)

def correlated_alias(m,rho):
 a,b=np.diag(m['F']);k=m['P']@np.ones(2);A=1-a*a;B=1-b*b;C=1-a*b;t=rho*np.sqrt(A*B)/C;t2=t*t
 if abs(t)<1e-14:s=0.
 else:
  disc=np.sqrt(t2*t2*(k.sum())**2+4*(1-t2)*t2*k.prod())
  s=2*t2*k.prod()/(t2*k.sum()+disc) if t>0 else -(t2*k.sum()+disc)/(2*(1-t2))
 P=np.array([[k[0]-s,s],[s,k[1]-s]]);Q=P-m['F']@P@m['F'].T;assert np.linalg.eigvalsh(Q).min()>-1e-8
 return dict(m,Q=Q,P=P,rho=float(rho),kind='correlated_alias')

def nodes(n=17,positive=False):
 x,w=np.polynomial.legendre.leggauss(n);return ((x+1)*.9999/2 if positive else x*.9999),w/2

def abs_normal(x,s):
 safe=np.maximum(s,1e-15);z=x/safe;v=2*safe*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+x*(2*ndtr(z)-1);return np.where(s>1e-14,v,abs(x))

def marginal(mu,var,w,y):
 # Flattened evaluation points P by mixture components K. Allow proper atoms.
 var=np.maximum(var,0);sd=np.sqrt(var);positive=sd>1e-14;safe=np.maximum(sd,1e-15);res=y[:,None]-mu;ll=np.where(positive,-.5*(np.log(2*np.pi*safe**2)+(res/safe)**2),-np.inf);nll=-logsumexp(np.log(np.maximum(w,1e-300))+ll,1);mass=(w*((~positive)&(res==0))).sum(1);nll=np.where(mass>0,-np.log(np.maximum(mass,1e-300)),nll);term=(w*abs_normal(res,sd)).sum(1);pair=np.zeros(len(mu))
 for i in range(mu.shape[1]):pair+=w[:,i]*np.sum(w*abs_normal(mu-mu[:,i,None],np.sqrt(var+var[:,i,None])),1)
 crps=term-.5*pair;mean=(w*mu).sum(1);v=(w*(var+mu*mu)).sum(1)-mean**2;out=dict(NLL=nll,CRPS=crps,MSE=(mean-y)**2,variance=np.maximum(v,0));lo=mu.min(1)-9*sd.max(1);hi=mu.max(1)+9*sd.max(1)
 # All eight quantiles in one vectorized bisection, including atomic components.
 qs=np.array([(1-l)/2 for l in LEVELS]+[(1+l)/2 for l in LEVELS]);left=np.broadcast_to(lo[:,None],(len(mu),8)).copy();right=np.broadcast_to(hi[:,None],left.shape).copy()
 for j in range(28):
  mid=(left+right)/2;cdf=np.sum(w[:,None]*np.where(positive[:,None],ndtr((mid[:,:,None]-mu[:,None])/safe[:,None]),(mid[:,:,None]>=mu[:,None]).astype(float)),2);left=np.where(cdf<qs,mid,left);right=np.where(cdf>=qs,mid,right)
 quant=(left+right)/2
 for i,level in enumerate(LEVELS):
  tag=str(int(level*100));out['cover'+tag]=((y>=quant[:,i])&(y<=quant[:,i+4])).astype(float);out['width'+tag]=quant[:,i+4]-quant[:,i]
 return out,mean,np.maximum(v,0)

def physical_scores(mu,cov,w,y):
 # mu P,K,2 ; cov P,K,2,2 ; weights P,K.
 P,K,d=mu.shape;parts=[];means=[];vars=[]
 for i in range(2):
  s,a,b=marginal(mu[:,:,i],cov[:,:,i,i],w,y[:,i]);parts.append(s);means.append(a);vars.append(b)
 result={k:(parts[0][k]+parts[1][k])/2 for k in parts[0]};inv=np.linalg.inv(cov);sign,det=np.linalg.slogdet(cov);assert np.all(sign>0);r=y[:,None]-mu;ll=-.5*(np.einsum('pki,pkij,pkj->pk',r,inv,r)+det+2*np.log(2*np.pi));lw=np.log(np.maximum(w,1e-300));result['joint_NLL']=-logsumexp(lw+ll,1)
 v1=np.maximum(cov[:,:,0,0],1e-14);z1=(y[:,0,None]-mu[:,:,0])/np.sqrt(v1);u1=(w*ndtr(z1)).sum(1);l1=-.5*(np.log(2*np.pi*v1)+z1*z1);conditional_w=np.exp(lw+l1-logsumexp(lw+l1,1)[:,None]);conditional_mu=mu[:,:,1]+cov[:,:,1,0]/v1*(y[:,0,None]-mu[:,:,0]);conditional_v=np.maximum(cov[:,:,1,1]-cov[:,:,1,0]**2/v1,1e-14);u2=(conditional_w*ndtr((y[:,1,None]-conditional_mu)/np.sqrt(conditional_v))).sum(1);radius=norm.ppf(np.clip(u1,1e-12,1-1e-12))**2+norm.ppf(np.clip(u2,1e-12,1-1e-12))**2
 for level in LEVELS:result['joint_cover'+str(int(level*100))]=(radius<=chi2.ppf(level,2)).astype(float)
 means=np.stack(means,1);between=(w*(cov[:,:,0,1]+mu[:,:,0]*mu[:,:,1])).sum(1)-means[:,0]*means[:,1];result['posterior_corr']=between/np.sqrt(np.maximum(np.array(vars).prod(0),1e-20));return result
