from pathlib import Path
import shutil,json,hashlib,datetime
R=Path(__file__).parent;S=Path('/workspace/causal/experiments/L1_hidden_representation');O=R/'inputs';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for seed in [101,202,303]:
 d=O/f'seed{seed}';d.mkdir(exist_ok=True)
 for f in ['X_only.npz','sealed_truth.npz','frozen.pt','probe_maps.json','FREEZE.json']:
  src=S/'results'/f'seed{seed}'/f;dst=d/f;shutil.copy2(src,dst);assert sha(src)==sha(dst);files[str(dst.relative_to(R))]=sha(dst)
for f in ['PROTOCOL.json','REGISTERED.json','run.py','REPORT.md']:
 dst=O/('L1_'+f);shutil.copy2(S/f,dst);files[str(dst.relative_to(R))]=sha(dst)
shutil.copy2(S/'results/trajectory_metrics.csv.gz',O/'L1_trajectory_metrics.csv.gz');files['inputs/L1_trajectory_metrics.csv.gz']=sha(O/'L1_trajectory_metrics.csv.gz')
(R/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(source_repo='https://github.com/lmo432185-lang/causal',source_commit='a033238a17ccab47c417ea2f03e6da4f5bf0f9e6',source_directory='experiments/L1_hidden_representation',time=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2))
print('Original L1 files copied and hash verified:',len(files))
