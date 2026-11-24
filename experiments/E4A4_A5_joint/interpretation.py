"""Model-coordinate interpretation; no generator labels or true parameters used."""
import json,gzip,time,numpy as np,pandas as pd
from settings import *
from model import unpack

def main():
 rows=[]
 for file in OUT.glob('*.json.gz'):
  with gzip.open(file,'rt') as f:b=unpack(json.load(f))
  for key,m in b['models'].items():
   d=len(m['C']);O=np.stack([m['C']@np.linalg.matrix_power(m['F'],k) for k in range(d)]) if d else np.empty((0,0));rank=int(np.linalg.matrix_rank(O,tol=1e-6)) if d else 0;poles=np.linalg.eigvals(m['F']) if d else np.array([])
   rows.append(dict(case=b['case'],key=key,physical_factor_count='unidentified',latent_Markov_dimension=d,observation_AR_order=int(abs(m.get('a',0))>1e-10),algebraic_observability_rank=rank,statistically_confirmed_mode_count='not established by matrix rank',legacy_factor_field=m.get('factor_count'),legacy_field_deprecated=m['kind']=='extension',pole_real=json.dumps(poles.real.tolist()),pole_imag=json.dumps(poles.imag.tolist()),equivalence_class='two_AR1_vs_one_ARMA21' if m['kind']=='factor2' else 'state_similarity_and_scalar_Gaussian_realizations',physical_identity='refuse_unique'))
 pd.DataFrame(rows).to_csv(OUT/'interpretation.csv.gz',index=False,compression='gzip');(ROOT/'INTERPRETATION_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),uses_true_system_labels=False,physical_count_not_identified=True,rows=len(rows)),indent=2))
if __name__=='__main__':main()
