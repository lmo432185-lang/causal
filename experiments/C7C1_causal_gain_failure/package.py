from pathlib import Path
import shutil,json,hashlib
ROOT=Path(__file__).parent;DEST=Path('/workspace/causal/experiments/C7C1_causal_gain_failure')
def main():
 DEST.mkdir(parents=True,exist_ok=True);r=DEST/'results';r.mkdir(exist_ok=True)
 for p in ROOT.iterdir():
  if p.is_file() and p.name!='bench.py' and (p.suffix in ['.py','.md','.json','.txt'] or p.name in ['run.log','analysis.log']):shutil.copy2(p,DEST/p.name)
 for p in (ROOT/'results').iterdir():
  if p.is_file():shutil.copy2(p,r/p.name)
 m={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts};(DEST/'MANIFEST.json').write_text(json.dumps(m,indent=2));print('files',len(m)+1,'MB',round(sum((DEST/f).stat().st_size for f in m)/1e6,2))
if __name__=='__main__':main()
