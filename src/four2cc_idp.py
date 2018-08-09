import numpy as np
import pandas as pd
from time import sleep
from pprint import pprint
from os import listdir
from os.path import isfile, join

df = pd.read_csv('../data_3_all_num_mat_tp.csv',header=0, index_col=0)
df['sum'] = df.sum(axis=1, numeric_only=True)

for i in df.index:
  if df.loc[i]['sum'] == 0:
    df.drop(i, inplace=True)
df.drop('sum', axis=1, inplace=True)

print("Calculating correlation...")
corr = {}
for k in df.index:
  if k == 'dx.1ad.2mci.3smi.x': continue
  print("Processing {}".format(k))
  corr[k] = np.absolute(np.corrcoef(df.loc['dx.1ad.2mci.3smi.x',:], df.loc[k,:]))[0,1] 

keys = list(corr.keys())
for k in keys:
  if np.isnan(corr[k]):
    corr.pop(k, None)

values = list(corr.values())
srtd_ix = np.argsort(values)

#pprint(df.loc[ [ keys[fid] for fid in srtd_ix[0:99] ],:])
print("Exporting to CSV...")
df.loc[ [ keys[fid] for fid in srtd_ix[-100:] ],:].to_csv('../100_final_data_idp.csv')
print(values[srtd_ix[-100]])
