import json,hashlib,inspect
import numpy as np
from scipy.special import logsumexp
from config import *
from generate import normal,equivalent_pair
from mechanisms import library,active_library
from inference import evidence,active_evidence
from legacy_stream import structured_scores
from warning_detectors import generic_scores

def main():
 ss=json.loads((OUT/'trained_settings.json').read_text())[1];bg=ss['bg'];x=normal(4,.35,217000);prefix=200
 a=structured_scores(x,bg,0)['M3'][:,:prefix//BLOCK];b=structured_scores(x[:,:prefix+1],bg,0)['M3'];e1=float(np.max(abs(a-b)));assert e1<1e-8
 a=generic_scores(x,bg)['GLR_Ramp'][:,:prefix//BLOCK];b=generic_scores(x[:,:prefix+1],bg)['GLR_Ramp'];e2=float(np.max(abs(a-b)));assert e2<1e-8
 bank=library(bg);bank={'A':bank['A_first'],'E':dict(bank['A_first'],y=bank['A_first']['u'])};a,b=equivalent_pair(4,.35,1.,217001)
 e=evidence(a['x'],bank,bg,ss['amp_prior']);assert np.array_equal(e[:,0],e[:,1]);ev=evidence(a['x'],bank,bg,ss['amp_prior'],a['U']+.5*a['eta'],ss['sensor_variances'][1]);assert np.array_equal(ev[:,0],ev[:,1])
 abank=active_library(bg);hidden=active_evidence(x,abank,bg);each=np.stack([active_evidence(x,abank,bg,np.full(len(x),i)) for i in range(len(INPUT_DESIGNS))],0);delta=float(np.max(abs(hidden-(logsumexp(each,axis=0)-np.log(len(INPUT_DESIGNS))))));assert delta<1e-9
 marker=json.loads((OUT/'TEST_START.json').read_text());unchanged={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in marker['sha256'].items()};assert all(unchanged.values())
 result=dict(M3_no_future_prefix_error=e1,GLR_no_future_prefix_error=e2,passive_equivalence_X_and_VU_likelihood_equal=True,hidden_input_exactly_marginalizes_known_protocol_likelihood_error=delta,frozen_files_unchanged=unchanged,inference_inputs=str(inspect.signature(evidence)),active_inference_inputs=str(inspect.signature(active_evidence)))
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
