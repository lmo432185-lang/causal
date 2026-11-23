from pathlib import Path
ROOT=Path(__file__).parent;DATA=ROOT/'data';OUT=ROOT/'results'
SEEDS=[101,202,303];N=800;T=500
P=20;L=20;FULL=P+L;H=10;KMAX=6
SPLITS={'initial':200,'recursive':200,'direct':200,'selection':50,'confirmation':50,'test':100}
START=FULL;END=T-H;LEVELS=[.5,.8,.9,.95]
MIN_GAIN=.005;MIN_PARTIAL_CORR=.5;MIN_RESIDUAL_FRACTION=.05
