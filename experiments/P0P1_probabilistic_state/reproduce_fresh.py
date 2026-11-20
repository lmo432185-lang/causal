"""Reproduce in a NEW directory. Never overwrite the published frozen experiment."""
import argparse,os,shutil,subprocess,sys,json,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--destination',required=True);a=p.parse_args();source=Path(__file__).parent;target=Path(a.destination).resolve()
if target.exists():raise SystemExit('Destination must not exist; frozen records must be preserved')
target.mkdir(parents=True)
for f in source.glob('*.py'):shutil.copy2(f,target/f.name)
shutil.copy2(source/'native_mean.cpp',target/'native_mean.cpp')
for directory in ['legacy','fonts']:shutil.copytree(source/directory,target/directory)
(target/'data').mkdir();(target/'results').mkdir();env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(target/'mpl_cache'))
def run(f):subprocess.run([sys.executable,str(target/f)],env=env,check=True)
for f in ['validate_dev.py','register.py','private_generator.py','fit_worker.py']:run(f)
files=['evaluate.py','PROTOCOL.json'];(target/'EVALUATION_REGISTERED.json').write_text(json.dumps(dict(time=time.time(),files={f:hashlib.sha256((target/f).read_bytes()).hexdigest() for f in files},thresholds='fixed in original protocol before generation',private_test_truth_not_opened=True),indent=2))
for f in ['evaluate.py','theory_controls.py','analytical_map.py','analyze.py','plot.py','build_report.py','audit.py']:run(f)
print('Completed fresh reproduction',target)
