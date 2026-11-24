import json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from producer import jobs,generate,digest
from settings import *
def one(j):
 x=generate(j['index'],j['seed'],0,70);return j['case'],{n:digest(a) for n,a in [('R',x[:20]),('D',x[20:40]),('selection',x[40:50])]}
if __name__=='__main__':
 rows={}
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(one,j) for j in jobs()]):k,h=f.result();rows[k]=h
 (ROOT/'TRAINING_DATA_FROZEN.json').write_text(json.dumps(dict(time=time.time(),cases=len(rows),hashes=rows,truth_not_opened=True),indent=2));(ROOT/'jobs.json').write_text(json.dumps(jobs(),indent=2));print('Training X frozen',len(rows))
