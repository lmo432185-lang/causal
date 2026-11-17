extern "C" void forward_mean(int N,int T,int d,int m,const double* y,const double* A,const double* H,const double* K,double* out){
 for(int n=0;n<N;n++){
  double prev[8]={0},pred[8],er[8],next[8];
  for(int t=0;t<T;t++){
   for(int i=0;i<d;i++){double q=0;for(int j=0;j<d;j++)q+=A[i*d+j]*prev[j];pred[i]=q;}
   for(int k=0;k<m;k++){double q=0;for(int j=0;j<d;j++)q+=H[k*d+j]*pred[j];er[k]=y[(n*T+t)*m+k]-q;}
   for(int i=0;i<d;i++){double q=pred[i];for(int k=0;k<m;k++)q+=K[(t*d+i)*m+k]*er[k];next[i]=q;out[(n*T+t)*d+i]=q;}
   for(int i=0;i<d;i++)prev[i]=next[i];
  }
 }
}
extern "C" void smooth_mean(int N,int T,int d,const double* f,const double* A,const double* J,double* out){
 for(int n=0;n<N;n++){
  for(int i=0;i<d;i++)out[(n*T+T-1)*d+i]=f[(n*T+T-1)*d+i];
  for(int t=T-2;t>=0;t--){double delta[8];
   for(int i=0;i<d;i++){double q=0;for(int j=0;j<d;j++)q+=A[i*d+j]*f[(n*T+t)*d+j];delta[i]=out[(n*T+t+1)*d+i]-q;}
   for(int i=0;i<d;i++){double q=f[(n*T+t)*d+i];for(int j=0;j<d;j++)q+=J[(t*d+i)*d+j]*delta[j];out[(n*T+t)*d+i]=q;}
  }
 }
}
