import hashlib,json,time
from config import *
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 (R/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2));files=['ssm.py','settings.py','config.py','generate.py','fit_jobs.py','equivalence.py','prepare.py','PROTOCOL.json','validate_dev.py','native_mean.py','native_mean.cpp','ssm_reference.py'];(R/'REGISTERED.json').write_text(json.dumps(dict(time=time.time(),files={f:sha(R/f) for f in files},development_validation='unrelated random arrays only; no experiment outcomes used'),indent=2));print('FROZEN',len(CELLS)*len(SEEDS),'datasets',flush=True)
