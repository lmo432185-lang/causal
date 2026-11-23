from pathlib import Path
ROOT=Path(__file__).parent;DATA=ROOT/'data';OUT=ROOT/'results'
SEEDS=list(range(1011,1021));SIZES=[100,200,400]
P=20;L=20;FULL=40;H=10;KMAX=6;T=500;START=40;END=490
METHODS=['recursive','direct_full','direct_fixed','direct_random']
CPU_LIMIT=60;MIN_GAIN=.005
