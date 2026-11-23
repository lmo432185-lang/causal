#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
const int M=3;
void zero(double*a,int n){fill(a,a+n,0.);}
void mm(const double*a,const double*b,double*c,int k,bool tb=false){for(int i=0;i<k;i++)for(int j=0;j<k;j++){double s=0;for(int l=0;l<k;l++)s+=a[i*k+l]*(tb?b[j*k+l]:b[l*k+j]);c[i*k+j]=s;}}
void inv(const double*a,double*out,int k){double b[18]={0};for(int i=0;i<k;i++)for(int j=0;j<2*k;j++)b[i*2*k+j]=j<k?a[i*k+j]:(j-k==i);for(int i=0;i<k;i++){double d=b[i*2*k+i];if(fabs(d)<1e-12)d=1e-12;for(int j=0;j<2*k;j++)b[i*2*k+j]/=d;for(int l=0;l<k;l++)if(l!=i){double v=b[l*2*k+i];for(int j=0;j<2*k;j++)b[l*2*k+j]-=v*b[i*2*k+j];}}for(int i=0;i<k;i++)for(int j=0;j<k;j++)out[i*k+j]=b[i*2*k+j+k];}
void chol(const double*a,double*l,int k){zero(l,k*k);for(int i=0;i<k;i++)for(int j=0;j<=i;j++){double s=.5*(a[i*k+j]+a[j*k+i]);for(int n=0;n<j;n++)s-=l[i*k+n]*l[j*k+n];l[i*k+j]=i==j?sqrt(max(s,1e-12)):s/l[j*k+j];}}
extern "C" void filter(const double*x,int n,int tmax,int k,const double*F,const double*Q,const double*C,double a,double R,const double*P0,int stop,int smooth,double*mf,double*pf,double*ms,double*ps,double*lag,double*ll,double*Jout,double*Lout){
 vector<double> pp(tmax*k*k),gain(tmax*k);double tmp[9],pred[9],vc[3],V;
 for(int t=0;t<tmax;t++){
  if(t==0)copy(P0,P0+k*k,pred);else {mm(F,pf+(t-1)*k*k,tmp,k);mm(tmp,F,pred,k,true);for(int j=0;j<k*k;j++)pred[j]+=Q[j];}
  copy(pred,pred+k*k,pp.data()+t*k*k);V=R;for(int i=0;i<k;i++){vc[i]=0;for(int j=0;j<k;j++)vc[i]+=pred[i*k+j]*C[j];V+=C[i]*vc[i];gain[t*k+i]=(t<stop?vc[i]/max(V,1e-12):0);}
  // V must include all coordinates before normalizing gain.
  for(int i=0;i<k;i++)gain[t*k+i]=(t<stop?vc[i]/max(V,1e-12):0);
  for(int i=0;i<k;i++)for(int j=0;j<k;j++)pf[t*k*k+i*k+j]=pred[i*k+j]-gain[t*k+i]*vc[j];
  for(int r=0;r<n;r++){
   double mp[3]={0};if(t)for(int i=0;i<k;i++)for(int j=0;j<k;j++)mp[i]+=F[i*k+j]*mf[(r*tmax+t-1)*k+j];
   double e=x[r*tmax+t]-(t?a*x[r*tmax+t-1]:0);for(int i=0;i<k;i++)e-=C[i]*mp[i];ll[r*tmax+t]=-.5*(log(2*M_PI*max(V,1e-12))+e*e/max(V,1e-12));for(int i=0;i<k;i++)mf[(r*tmax+t)*k+i]=mp[i]+gain[t*k+i]*e;
  }
 }
 if(!smooth)return;copy(mf,mf+n*tmax*k,ms);copy(pf,pf+tmax*k*k,ps);
 for(int t=tmax-2;t>=0;t--){double iv[9],jmat[9],d[9],z[9],S[9];inv(pp.data()+(t+1)*k*k,iv,k);mm(pf+t*k*k,F,tmp,k,true);mm(tmp,iv,jmat,k);copy(jmat,jmat+k*k,Jout+t*k*k);for(int i=0;i<k*k;i++)d[i]=ps[(t+1)*k*k+i]-pp[(t+1)*k*k+i];mm(jmat,d,tmp,k);mm(tmp,jmat,z,k,true);for(int i=0;i<k*k;i++)ps[t*k*k+i]=pf[t*k*k+i]+z[i];mm(ps+(t+1)*k*k,jmat,lag+t*k*k,k,true);
  mm(jmat,pp.data()+(t+1)*k*k,tmp,k);mm(tmp,jmat,z,k,true);for(int i=0;i<k*k;i++)S[i]=pf[t*k*k+i]-z[i];chol(S,Lout+t*k*k,k);
  for(int r=0;r<n;r++){double v[3]={0};for(int i=0;i<k;i++){v[i]=ms[(r*tmax+t+1)*k+i];for(int j=0;j<k;j++)v[i]-=F[i*k+j]*mf[(r*tmax+t)*k+j];}for(int i=0;i<k;i++)for(int j=0;j<k;j++)ms[(r*tmax+t)*k+i]+=jmat[i*k+j]*v[j];}
 }
 chol(pf+(tmax-1)*k*k,Lout+(tmax-1)*k*k,k);
}
extern "C" void draw(const double*mf,const double*F,const double*J,const double*L,const double*noise,int draws,int n,int T,int k,double*out){
 for(int d=0;d<draws;d++)for(int r=0;r<n;r++)for(int t=T-1;t>=0;t--){double v[3]={0};if(t<T-1)for(int i=0;i<k;i++){v[i]=out[((d*n+r)*T+t+1)*k+i];for(int j=0;j<k;j++)v[i]-=F[i*k+j]*mf[(r*T+t)*k+j];}for(int i=0;i<k;i++){double m=mf[(r*T+t)*k+i];if(t<T-1)for(int j=0;j<k;j++)m+=J[t*k*k+i*k+j]*v[j];for(int j=0;j<k;j++)m+=L[t*k*k+i*k+j]*noise[((d*n+r)*T+t)*k+j];out[((d*n+r)*T+t)*k+i]=m;}}
}
