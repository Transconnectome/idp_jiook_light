import numpy as np
import pandas as pd
from pprint import pprint
from os import listdir
from os.path import isfile, join

print("Processing freesufer data...")
freesurf = '../data/freesurfer'
fs_lst = [f for f in listdir(freesurf) if isfile(join(freesurf, f))]
df_lst = []
for f in fs_lst:
  print("Procesing : {}/{}".format(freesurf,f))
  df_lst.append(pd.read_csv(join(freesurf, f), sep='\s+', header=0, index_col=0).transpose())


print("Processing mrtrix data...")
mrtrix = "../data/data_mrtrix.csv"
df_mrtrix = (pd.read_csv(mrtrix, sep=',', header=0, index_col=1).transpose())

df = pd.concat(df_lst) # contains all freesurfer
df.columns = [int(c[:-1]) for c in list(df)] # trim trailing  / in column ids

df = pd.concat([df_mrtrix, df])

subj_ids = list(df_mrtrix)
df = df[subj_ids] # trim down freesurfer data

df.drop(df_mrtrix.index [:-1], inplace=True)
df.dropna(inplace=True)

mrm = pd.DataFrame({})
mrm_lst = [df]
q = 0
for s in subj_ids:
  rh = []
  data = []
  for i in ['1_count_volumeadj', '1_length']: # 2 is too  big
  #for i in ['1_count', '1_count_volumeadj', '1_length', '2_count', '2_count_volumeadj', '2_length']:
    print("Procesing : data/matrix/{}_map{}".format(s,i))
    mr = (pd.read_csv("../data/matrix/{}_map{}.csv".format(s,i), sep='\s+', header=None))
    rh += [ "{}_{}".format(i,e) for e in range(mr.size)] # row headings
    data += mr.as_matrix().reshape(1,mr.size)[0].tolist()
  pprint (len(rh))
  mr2= pd.DataFrame({s : data}, index=rh, columns=[s])
  mrm[s] = mr2[s]
df = pd.concat([df, mrm])
df.to_csv('../all_data.csv')
