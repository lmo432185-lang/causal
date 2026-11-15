from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).parent;DEST=Path('/workspace/causal/experiments/C7B2_input_design')
def main():
 DEST.mkdir(parents=True,exist_ok=True);r=DEST/'results';r.mkdir(exist_ok=True)
 for p in ROOT.iterdir():
  if p.is_file() and (p.suffix in ['.py','.md','.json','.txt'] or p.name in ['run.log','analysis.log','confirmation.log']):shutil.copy2(p,DEST/p.name)
 for p in (ROOT/'results').iterdir():
  if p.is_file():shutil.copy2(p,r/p.name)
 m={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts};(DEST/'MANIFEST.json').write_text(json.dumps(m,indent=2));print('Files',len(m)+1,'MB',round(sum((DEST/p).stat().st_size for p in m)/1e6,2))
if __name__=='__main__':main()
