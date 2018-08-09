import numpy as np
import pandas as pd
from pprint import pprint

df_lh = pd.read_csv('aparc_area_lh.txt', sep='\s+', header=0, index_col=0).transpose()
df_rh = pd.read_csv('aparc_area_rh.txt', sep='\s+', header=0, index_col=0).transpose()
df = pd.concat([df_lh, df_rh])
pprint(df.iloc[1,:])
pprint(df.iloc[:,1])
pprint(df.transpose().corr())
