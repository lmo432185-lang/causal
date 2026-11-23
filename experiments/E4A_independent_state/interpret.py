"""Post-fit interpretation certificates. No family truth labels accepted."""
import json,time,numpy as np
from settings import *
from model import unpack,pack
from producer import arma_alias

def interpret(bundle):
 result={'physical_identity':'not_uniquely_established_from_scalar_X','scope':'working Gaussian family preference only; no physical factor-count posterior','certificates':[]}
 for route in ['direct','recursive','witness']:
  m=bundle[route]['models']['factor2']
  if m is None:continue
  alt=arma_alias(m);result['certificates'].append({'route':route,'working_interpretation':'two_independent_AR1_contributions','equivalent_interpretation':'one_scalar_ARMA21_contribution','same_X_law':True,'unique_physical_factor_count':'refuse','alias_parameters':pack(alt),'posterior_odds_note':'identical likelihood keeps prior odds for mapped parameter priors; no parameter integration claimed'})
 return result

def main():
 assert (ROOT/'TEST_B_FROZEN.json').exists()
 n=0
 for p in OUT.glob('*/model_*.json'):
  b=unpack(json.loads(p.read_text()));r=interpret(b);(p.parent/(p.stem.replace('model','interpretation')+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2));n+=1
 (ROOT/'INTERPRETATION_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),bundles=n,uses_system_label=False,equivalence_refusal_is_analytic_not_learned=True),indent=2));print('Analytic interpretation certificates',n)
if __name__=='__main__':main()
