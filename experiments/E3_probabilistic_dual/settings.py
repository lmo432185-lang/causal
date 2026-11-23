from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
SEEDS=list(range(3011,3031));SIZES=[20,50,100,200,400];T=160;H=20;P=40;KMAX=3
ORIGINS=list(range(40,131,10));HORIZONS=[1,2,5,10,20];GROUPS=['A','B','C','D','E','F','G'];EM_ITERS=5
