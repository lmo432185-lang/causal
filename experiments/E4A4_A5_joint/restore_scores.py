"""Restore original CSV.gz scores exactly, or verify reconstruction without writing.

python restore_scores.py --verify-only
python restore_scores.py
"""
import json,lzma,hashlib,sys
from pathlib import Path
from compact_scores import reconstruct
ROOT=Path(__file__).parent
def main():
 record=json.loads((ROOT/'compact/INDEX.json').read_text());verify='--verify-only' in sys.argv
 for rec in record['files']:
  raw=lzma.decompress((ROOT/rec['payload']).read_bytes());assert hashlib.sha256(raw).hexdigest()==rec['raw_sha256'];data=reconstruct(raw,rec['header'],rec['trailer'],rec['sync_flushes']);assert hashlib.sha256(data).hexdigest()==rec['sha256'];path=ROOT/rec['path']
  if not verify:
   path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
   else:path.write_bytes(data)
  print('VERIFIED' if verify else 'RESTORED',rec['path'])
 print('All original compressed file SHA256 values match; no experiment recomputation')
if __name__=='__main__':main()
