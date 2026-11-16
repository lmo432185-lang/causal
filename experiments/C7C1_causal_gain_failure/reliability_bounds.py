"""Additional simultaneous normal-FPR audit, no threshold/policy changes."""
import pandas as pd
from scipy.stats import beta
from config import OUT

def main():
 d=pd.read_csv(OUT/'normal_audit.csv');alpha=.05/len(d);d['simultaneous_one_sided_upper95']=beta.ppf(1-alpha,d['count']+1,d['n']-d['count']);d.to_csv(OUT/'normal_simultaneous_bounds.csv',index=False);print(d.groupby('budget').simultaneous_one_sided_upper95.max().to_dict())
if __name__=='__main__':main()
