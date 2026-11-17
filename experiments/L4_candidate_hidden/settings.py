# Public inference settings only. Does not import generator or hidden truth manifest.
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
NSTART=4;MAXITER=40;START=100;CONF=.95
