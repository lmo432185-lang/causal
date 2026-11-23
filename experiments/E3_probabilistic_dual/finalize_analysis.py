"""Pretest-specified safety/count supplement; preserve original analysis outputs."""
import json,pandas as pd
from settings import *

def main():
 cf=pd.read_csv(OUT/'confirmation.csv');g=cf.groupby(['case','seed','n','route','K']).agg(both_supported=('supported','all'),gain=('gain','mean')).reset_index();g['condition']=g.case.str.rsplit('_',n=1).str[0];g.to_csv(OUT/'support_by_seed.csv',index=False);error=[]
 for cond,k in [('S0',1),('S1',2)]:
  z=g[(g.condition==cond)&(g.K==k)&(g.n==100)];p=z.pivot(index='seed',columns='route',values='both_supported');error.append(dict(condition=cond,direct_errors=int(p.direct.sum()),recursive_errors=int(p.recursive.sum()),seeds=20,excess=int(p.recursive.sum()-p.direct.sum())))
 con=json.loads((OUT/'CONCLUSION.json').read_text());con['false_expansion_diagnostics']=error;con['safety_no_large_observed_increase']=all(r['excess']<3 for r in error);con['unique_synergy_demonstrated']=bool(con['unique_synergy_demonstrated'] and con['safety_no_large_observed_increase']);con['safety_rule_note']='pretest supplement: >=3/20 excess confirmed errors fails guard; smaller differences do not establish equal true error rates';(OUT/'FINAL_CONCLUSION.json').write_text(json.dumps(con,indent=2));print(json.dumps(error,indent=2))
if __name__=='__main__':main()
