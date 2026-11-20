"""Equivalent exact Gaussian mean recurrences; optional C++ acceleration."""
import ctypes,hashlib,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;available=False;lib=None
if os.environ.get('L4_NATIVE','1')!='0':
 try:
  src=ROOT/'native_mean.cpp';cache=ROOT/'.native_cache';cache.mkdir(exist_ok=True);so=cache/(hashlib.sha256(src.read_bytes()).hexdigest()[:16]+'.so')
  if not so.exists():subprocess.run(['g++','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(src),'-o',str(so)],check=True,capture_output=True)
  lib=ctypes.CDLL(str(so));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.forward_mean.argtypes=[ctypes.c_int]*4+[ptr]*5;lib.smooth_mean.argtypes=[ctypes.c_int]*3+[ptr]*4;available=True
 except (OSError,subprocess.CalledProcessError):pass

def means(y,A,H,K,J,smooth=True):
 y,A,H,K,J=[np.ascontiguousarray(v,dtype='float64') for v in [y,A,H,K,J]];N,T,m=y.shape;d=len(A);assert d<=8 and m<=8;f=np.empty((N,T,d));lib.forward_mean(N,T,d,m,y,A,H,K,f)
 if smooth:s=np.empty_like(f);lib.smooth_mean(N,T,d,f,A,J,s)
 else:s=None
 return f,s
