#include <cmath>
#include <vector>
#include <algorithm>
extern "C" double ll2(const double*x,int n,int T,double a,double b,double k1,double k2,double R){
 std::vector<double> m1(n,0.),m2(n,0.);double p1=k1,p2=k2,p12=0.,out=0.;double q1=k1*(1-a*a),q2=k2*(1-b*b);
 for(int t=0;t<T;t++){
  if(t){p1=a*a*p1+q1;p2=b*b*p2+q2;p12=a*b*p12;}
  double v1=p1+p12,v2=p2+p12,S=std::max(v1+v2+R,1e-12),g1=v1/S,g2=v2/S,e2=0.;
  for(int i=0;i<n;i++){double u=t?a*m1[i]:0.,v=t?b*m2[i]:0.,e=x[i*T+t]-u-v;m1[i]=u+g1*e;m2[i]=v+g2*e;e2+=e*e;}
  out-=.5*(n*std::log(2*M_PI*S)+e2/S);p1-=g1*v1;p2-=g2*v2;p12-=g1*v2;
 }
 return out;
}
