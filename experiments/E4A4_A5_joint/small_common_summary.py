"""Matched C5 versus frozen main-state summaries; descriptive only, no testing."""
import pandas as pd
from settings import ROOT,OUT
def main():
 d=pd.read_csv(OUT/'signal_probability.csv.gz');d['condition']=d.case.str.split('_').str[0];d=d[d.condition.isin(['CL','CM','CH','CE'])&d.seed.isin([4011,4012,4013])&(d.trajectory<5)];d.groupby(['batch','condition','method']).mean(numeric_only=True).reset_index().to_csv(OUT/'small_common_signal_summary.csv',index=False)
if __name__=='__main__':main()
