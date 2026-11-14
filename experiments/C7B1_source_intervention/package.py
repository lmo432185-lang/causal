from pathlib import Path
import shutil,hashlib,json
import numpy as np
ROOT=Path(__file__).parent;DEST=Path('/workspace/causal/experiments/C7B1_source_intervention')
def main():
 DEST.mkdir(parents=True,exist_ok=True);r=DEST/'results';r.mkdir(exist_ok=True)
 for p in ROOT.iterdir():
  if p.is_file() and (p.suffix in ['.py','.md','.json','.txt'] or p.name in ['run.log','analysis.log']):shutil.copy2(p,DEST/p.name)
 for p in (ROOT/'results').iterdir():
  if not p.is_file():continue
  if p.name.startswith(('calibration_scores_','null_scores_')):
   with np.load(p) as z:np.savez_compressed(r/p.name.replace('_scores_','_maxima_'),**{a:(np.minimum(z[a][:,1:],z[a][:,:-1]).max(1) if a in ['X','X_U'] else z[a]) for a in z.files})
  else:shutil.copy2(p,r/p.name)
 manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts};(DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('Packaged',len(manifest)+1,'files, MB',round(sum((DEST/p).stat().st_size for p in manifest)/1e6,2))
if __name__=='__main__':main()
