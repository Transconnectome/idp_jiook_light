import numpy as np
import pandas as pd
from pprint import pprint
from os import listdir
from os.path import isfile, join

freesurf = '../data/freesurfer'

fs_lst = [f for f in listdir(freesurf) if isfile(join(freesurf, f))]
df_lst = []
for f in fs_lst:
  print("Procesing : {}/{}".format(freesurf,f))
  df_lst.append(pd.read_csv(join(freesurf, f), sep='\s+', header=0, index_col=0).transpose())
df = pd.concat(df_lst)
pprint(df.iloc[1,:])
pprint(df.iloc[:,1])
pprint(df.transpose().corr())
