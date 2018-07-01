# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:35:51 2018

@author: bwhe
"""

import pickle
import pandas as pd


df = pd.read_csv('../data/train_list_song_info.csv.gz', usecols=['pid', 'track_uri'], nrows=None)


df = df.groupby('track_uri')['pid'].count().reset_index()
df = df.rename(columns={'pid': 'freq'})
df.to_csv('../data/songfreq.csv.gz', index=False, compression='gzip')

d = {}
for i, j in zip(df.track_uri, df.freq):
    d[i] = j
with open('../data/songfreq.pkl', 'wb') as f:
    pickle.dump(d, f)

