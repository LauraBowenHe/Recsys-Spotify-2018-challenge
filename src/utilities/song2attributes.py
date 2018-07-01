# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:45:16 2018

@author: bwhe
"""

import gc
import pandas as pd
from collections import defaultdict
import pickle

print("read song_info...")
df = pd.read_csv('../data/song_info.csv.gz')
#########################################################################################
# song to artist
#########################################################################################
print("create song to artist dic...")
d = {}
for i, j in zip(df.track_uri, df.artist_uri):
    d[i] = j

with open('../data/song2artist.pkl', 'wb') as f:
    pickle.dump(d, f)
del d
gc.collect()

#########################################################################################
# song to album
#########################################################################################
print("create song to album dic...")
d = {}
for i, j in zip(df.track_uri, df.album_uri):
    d[i] = j

with open('../data/song2album.pkl', 'wb') as f:
    pickle.dump(d, f)
del d
gc.collect()
    
#########################################################################################
# song to track name
#########################################################################################
print("song to track name dic...")
d = {}
for i, j in zip(df.track_uri, df.track_name):
    d[i] = j
with open('../data/song2names.pkl', 'wb') as f:
    pickle.dump(d, f)
del d
gc.collect()

#########################################################################################
# song to artist name
#########################################################################################
print("song to artist name dic...")
d = {}
for i, j in zip(df.track_uri, df.artist_name):
    d[i] = j
with open('../data/song2artist_name.pkl', 'wb') as f:
    pickle.dump(d, f)
del d
gc.collect()

#########################################################################################
# song to album name
#########################################################################################
print("song to album name dic...")
d = {}
for i, j in zip(df.track_uri, df.album_name):
    d[i] = j
with open('../data/song2album_name.pkl', 'wb') as f:
    pickle.dump(d, f)
del d
gc.collect()

#######################################################################################
# song label to song 
#######################################################################################
print("song label to track uri dic...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
with open('../data/le_track_uri.pkl', 'rb') as f:
    song = pickle.load(f)
label_song = le.fit_transform(song)
d = defaultdict(list)
for i, j in zip(label_song, song):
    d[i].append(j)
d = dict(d)

with open('../data/songlabel2uri.pkl', 'wb') as f:
    pickle.dump(d, f)


#''' check part '''
#with open('../data/test_pid2more_clean_name.pkl', 'rb') as f:
#    s2a = pickle.load(f)
#
#with open('../../test_pid2more_clean_name.pkl', 'rb') as f:
#    s2a_1 = pickle.load(f)
#
#for i in s2a_1.keys():
#    if s2a[i] != s2a_1[i]:
#        print("Oops, %i not match..."%(i))
#        print(s2a[i])
#        print(s2a_1[i])