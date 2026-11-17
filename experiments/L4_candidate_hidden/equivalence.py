"""Estimated-parameter stationary-law audit, never reads truth."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from ssm import mirror

def kernel(p,lags=101):
 A=p['A'];Q=p['Q'];P=solve_discrete_lyapunov(A,Q);ix=1 if p['kind']=='single' else 2;v=np.array([(np.linalg.matrix_power(A,k)@P)[ix,ix] for k in range(lags)]);v[0]+=p['r'];return v

def counterpart(p):
 kind=p['kind'];A=p['A'];a,c=A[0,0],A[1,1];b=A[2,2];g1,g2=A[2,0],A[2,1];ans=[]
 if kind=='G4':
  cp=mirror(p);return [dict(label='G2',relative_covariance_error=float(np.max(abs(kernel(cp)-kernel(p)))/max(np.max(abs(kernel(p))),1e-12)),parameters={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cp.items()})]
 if kind=='G2':ans.append(('G4',mirror(p)))
 def chain(source,middle,gain,alpha):
  aa=np.array([[source,0,0],[alpha,middle,0],[0,gain,b]]);return {**p,'A':aa,'Q':p['Q'][:3,:3].copy(),'P0':solve_discrete_lyapunov(aa,p['Q'][:3,:3]),'kind':'G2','delay':1}
 if kind=='G1':
  if a<c:a,c=c,a;g1,g2=g2,g1
  gain2=g2*g2+g1*g1*c/max(a,1e-12);alpha2=g1*g1*(a-c)*(1-a*c)/(max(a,1e-12)*max(gain2,1e-12))
  if gain2>1e-12 and alpha2>=-1e-10:
   cp=chain(a,c,np.sqrt(gain2),np.sqrt(max(alpha2,0)));ans.extend([('G2',cp),('G4',mirror(cp))])
 elif kind=='G2':
  alpha=A[1,0];den=(1+c*c)-(c/max(a,1e-12))*(1+a*a)
  if abs(den)>1e-10:
   v=alpha*alpha/den;w=1-c/max(a,1e-12)*v
   if v>=-1e-10 and w>=-1e-10:
    aa=np.array([[a,0.,0.],[0.,c,0.],[g2*np.sqrt(max(v,0)),g2*np.sqrt(max(w,0)),b]]);ans.append(('G1',{**p,'A':aa,'kind':'G1','delay':1,'P0':solve_discrete_lyapunov(aa,p['Q'])}))
 elif kind=='G3' and p['delay']==1:
  alpha=A[1,0];L=g2*g2*a+g1*g1*c-g1*g2*alpha;M=g1*g1+(g2*alpha-g1*c)**2+g2*g2*(1+a*a)
  for source,middle in [(a,c),(c,a)]:
   gain2=L/max(source,1e-12);alpha2=M/max(gain2,1e-12)-1-source*source
   if gain2>1e-12 and alpha2>=-1e-10:
    cp=chain(source,middle,np.sqrt(gain2),np.sqrt(max(alpha2,0)));ans.extend([('G2',cp),('G4',mirror(cp))]);break
 good=[];ref=kernel(p)
 for label,cp in ans:
  if np.max(abs(cp['A']))>3.000001:continue
  err=float(np.max(abs(kernel(cp)-ref))/max(np.max(abs(ref)),1e-12))
  if err<1e-8:good.append(dict(label=label,relative_covariance_error=err,parameters={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cp.items()}))
 return good
