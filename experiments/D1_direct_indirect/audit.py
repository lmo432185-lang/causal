import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).parent
checks={}
for name,key in [('results/TEST_START.json','sha256'),('TIMING_PROTOCOL.json','code_hashes')]:
 m=json.loads((R/name).read_text());bad=[f for f,h in m[key].items() if hashlib.sha256((R/f).read_bytes()).hexdigest()!=h];assert not bad,bad;checks[name]='all frozen hashes match'
d=pd.read_csv(R/'results/structure_state_metrics.csv.gz',keep_default_na=False,na_values=['']);assert len(d)==52200;assert not d.duplicated(['cell','cohort','trajectory']).any();assert ((d.correct_declared+d.wrong_declared+d.refused)==1).all();assert d.groupby('cell').size().eq(200).all();checks['structure_rows']=len(d)
t=pd.read_csv(R/'results/timing_metrics.csv.gz',keep_default_na=False,na_values=['']);assert len(t)==7200;assert (t[t['mode']=='beta_loss'].upstream_identical_to_null).all();assert t[(t.method=='SourceY_retrospective')].available_time.eq(32).all();checks['timing_rows']=len(t)
e=pd.read_csv(R/'results/equivalence_metrics.csv.gz');assert len(e)==6000;p=json.loads((R/'results/equivalence_proofs.json').read_text());assert max(v['relative_covariance_error'] for v in p)<1e-10;checks['exact_equivalence_covariance_max_error']=max(v['relative_covariance_error'] for v in p)
checks['non_online_RTS_available_at']=32;checks['status']='PASS';(R/'results/audit.json').write_text(json.dumps(checks,indent=2));print(checks)
