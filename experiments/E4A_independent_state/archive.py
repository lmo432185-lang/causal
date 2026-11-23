"""Copy completed audited experiment into its isolated repository branch."""
import json,time,hashlib,shutil,subprocess
from pathlib import Path
from settings import ROOT,OUT
DEST=Path('/workspace/causal/experiments/E4A_independent_state')
def main():
 for name in ['AUDIT_COMPLETE','TRUTH_AUDIT_COMPLETE','ANALYSIS_COMPLETE','ROTATION_AUDIT_COMPLETE','STATE_EXTRA_COMPLETE','REPORT_RENDERED']:
  assert (ROOT/(name+'.json')).exists(),name
 assert len(list((ROOT/'figures').glob('*.png')))==13
 for p in ROOT.glob('*REGISTERED.json'):
  for f,h in json.loads(p.read_text()).get('files',{}).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(p.name,f)
 DEST.mkdir(parents=True,exist_ok=True)
 for f in ROOT.iterdir():
  if f.is_file() and (f.suffix in ['.py','.cpp','.md','.json','.txt','.html','.pdf']) and f.name not in ['STATUS.json']:
   shutil.copy2(f,DEST/f.name)
 for name in ['fonts','figures']:
  shutil.copytree(ROOT/name,DEST/name,dirs_exist_ok=True)
 shutil.copytree(OUT,DEST/'results',dirs_exist_ok=True)
 # No redundant cache, checkpoints or raw trajectories are archived.
 (DEST/'ARCHIVE_SCOPE.json').write_text(json.dumps(dict(time=time.time(),scope='E4-A1/A2/A3',raw_data='Full data deterministically regenerable, input hashes saved; 40 fixed example X/posterior files included',excluded=['fit_checkpoints','native shared library','Python caches','runtime logs'],previous_E3_commit='da8dbc1ad788891e1addb8b67eeb52cf33578e4c',main_base='1a95f3cca31498045caff694a5aa58c6bfd01f86',source=str(ROOT)),ensure_ascii=False,indent=2))
 files={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'};(DEST/'MANIFEST_SHA256.json').write_text(json.dumps(files,indent=2));assert all(hashlib.sha256((DEST/f).read_bytes()).hexdigest()==h for f,h in files.items());print('Archived files',len(files),'bytes',sum((DEST/f).stat().st_size for f in files))
if __name__=='__main__':main()
