"""Archive completed artifacts without modifying frozen numerical files or old branches."""
import os,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).parent;DEST=Path('/workspace/causal/experiments/E4A4_A5_joint')
def main():
 assert (ROOT/'OUTPUTS_COMPLETE.json').exists()
 files=[]
 for p in ROOT.rglob('*'):
  if not p.is_file() or p.name=='MANIFEST.sha256':continue
  rel=p.relative_to(ROOT)
  if any(x in rel.parts for x in ['__pycache__','checkpoints','.git','fonts']):continue
  if p.suffix in ['.so','.tmp','.log'] or p.name=='completion.py':continue
  if rel.parts[0]=='short' and len(rel.parts)>2 and rel.parts[1] not in ['results']:continue
  # Full source, frozen metadata, compressed parameters and all completed scores.
  if p.suffix not in ['.py','.cpp','.md','.json','.gz','.xz','.csv','.npz','.png','.pdf','.html','.txt']:continue
  files.append((p,rel))
 for src,rel in files:
  dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   if hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest():continue
   dst.unlink()
  # Immutable large numerical files are hardlinked to avoid duplicating scarce storage.
  os.link(src,dst)
 # Preserve the small original lost-run log despite the generic *.log ignore rule.
 src=ROOT/'storage_incident/original_fit.log'
 if src.exists():shutil.copy2(src,DEST/'storage_incident/original_fit_log.txt')
 font=Path('/workspace/work/E4A_probabilistic/fonts');d=DEST/'fonts';d.mkdir(exist_ok=True)
 for name in ['LICENSE.txt']:
  target=d/name
  if not target.exists():os.link(font/name,target)
 metadata=dict(font='NotoSansCJKsc-Regular.otf',sha256=hashlib.sha256((font/'NotoSansCJKsc-Regular.otf').read_bytes()).hexdigest(),source='https://raw.githubusercontent.com/lmo432185-lang/causal/experiment/e4a-independent-state-20261123/experiments/E4A_independent_state/fonts/NotoSansCJKsc-Regular.otf',note='Chinese glyphs are embedded in REPORT.pdf and rasterized in PNGs. Restore this font for regenerating plots or offline HTML; license included.')
 (d/'FONT_SOURCE.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 # Exclude only this manifest itself and known runtime products, matching Git ignores.
 records=[]
 for p in sorted(DEST.rglob('*')):
  if not p.is_file() or p.name=='MANIFEST.sha256' or any(x in p.parts for x in ['__pycache__','checkpoints']):continue
  if p.suffix in ['.so','.log','.tmp']:continue
  records.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(DEST).as_posix())
 (DEST/'MANIFEST.sha256').write_text('\n'.join(records)+'\n')
 (ROOT/'ARCHIVE_COMPLETE.json').write_text(json.dumps(dict(files=len(records),destination=str(DEST)),indent=2));print('Archived',len(records),'files')
if __name__=='__main__':main()
