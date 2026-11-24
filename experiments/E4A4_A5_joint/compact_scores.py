"""Lossless storage encoding only: CSV.xz + exact original gzip reconstruction.

Large scores are compressed across a longer dictionary to fit the cloud disk.
No numerical value changes. Originals removed only after reconstruction SHA proof.
"""
import gzip,zlib,lzma,json,hashlib,os
from pathlib import Path
ROOT=Path(__file__).parent
def digest(b):return hashlib.sha256(b).hexdigest()
def header_size(b):
 flag=b[3];i=10
 if flag&4:i+=2+int.from_bytes(b[i:i+2],'little')
 if flag&8:i=b.index(b'\0',i)+1
 if flag&16:i=b.index(b'\0',i)+1
 if flag&2:i+=2
 return i
def reconstruct(raw,head,tail,sync):
 c=zlib.compressobj(9,zlib.DEFLATED,-15);body=c.compress(raw)
 for _ in range(sync):body+=c.flush(zlib.Z_SYNC_FLUSH)
 return bytes.fromhex(head)+body+c.flush()+bytes.fromhex(tail)
def main():
 assert (ROOT/'OUTPUTS_COMPLETE.json').exists()
 out=ROOT/'compact';out.mkdir(exist_ok=True);records=[];paths=[]
 for folder in [ROOT/'results',ROOT/'short/results']:
  for p in sorted(folder.glob('*.csv.gz')):
   if p.stat().st_size<1000000:continue
   b=p.read_bytes();raw=gzip.decompress(b);i=header_size(b);head=b[:i].hex();tail=b[-8:].hex();sync=next((n for n in range(3) if reconstruct(raw,head,tail,n)==b),None)
   if sync is None:print('KEEP: nonstandard gzip stream',p.name,flush=True);continue
   rel=p.relative_to(ROOT);payload=Path('compact')/Path(str(rel).removesuffix('.gz')+'.xz');dest=ROOT/payload;dest.parent.mkdir(parents=True,exist_ok=True);packed=lzma.compress(raw,preset=6)
   with dest.open('wb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
   restored=reconstruct(lzma.decompress(dest.read_bytes()),head,tail,sync);assert digest(restored)==digest(b)
   records.append(dict(path=rel.as_posix(),payload=payload.as_posix(),sha256=digest(b),raw_sha256=digest(raw),header=head,trailer=tail,sync_flushes=sync,level=9,original_bytes=len(b),stored_bytes=len(packed)));paths.append(p);print('PACKED',rel,len(b),len(packed),flush=True)
 index=dict(format='lossless CSV.xz with byte-identical gzip reconstruction',zlib_version=zlib.ZLIB_VERSION,model_or_metric_values_changed=False,files=records);tmp=out/'INDEX.tmp';tmp.write_text(json.dumps(index,indent=2));tmp.replace(out/'INDEX.json')
 for p in paths:p.unlink()
 print('Original task-generated score copies replaced with lossless payloads; saving',sum(x['original_bytes']-x['stored_bytes'] for x in records),flush=True)
if __name__=='__main__':main()
