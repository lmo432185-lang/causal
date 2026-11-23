from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'results'
SEEDS=list(range(4011,4031));LENGTHS=[1000,5000];TMAX=5000;H=20
HORIZONS=[1,2,5,10,20];KINDS=['obs_ar','factor1','memory2','memory3','factor2'];GROUPS=['M1','M2','M3','M4','Xhistory'];EM_MAX=80;TOL=1e-6
NTRAIN=20;NVAL=10;NCONF=10;NTEST=20

def origins(T):return list(range(200,T-H,100))
