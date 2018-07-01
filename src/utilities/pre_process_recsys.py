# -*- coding: utf-8 -*-


import json
import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from copy import deepcopy
from tqdm import tqdm

data_path = '../data/'
dirs = os.listdir( data_path + 'mpd.v1/data/' )

count = 0

print("Information Extraction from the Train Data:")

for json_file in tqdm(dirs):    
#    print(count)
    count = count + 1
    f = open(data_path+'mpd.v1/data/'+json_file)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)
    tmp_slice = pd.DataFrame.from_dict(mpd_slice['playlists'])
    tmp_slice = tmp_slice[['pid','tracks']]
    if count == 1:
        all_song = tmp_slice
    else:
        all_song = pd.concat((all_song,tmp_slice))
        
all_song = all_song.reset_index(drop=True)        
    
all_song_unfold = all_song['tracks'].apply(pd.Series).stack()

all_song_unfold_df = pd.DataFrame(all_song_unfold)
all_song_unfold_df['level_0'] = np.array(all_song_unfold.index.get_level_values(0))

cols = ['artist_uri','pos','artist_name','track_uri','track_name','album_uri','duration_ms','album_name']

for col in cols:
    all_song_unfold_df[col] = all_song_unfold.apply(lambda x:x[col])
    
all_song_unfold_df = all_song_unfold_df.drop([0],axis=1)

all_song_unfold_df = all_song_unfold_df.reset_index(drop=True)

all_song_unfold = []

gc.collect()

all_song['level_0'] = all_song.index.get_level_values(0)
all_song = all_song.drop(['tracks'],axis=1)

all_song_unfold_df = pd.merge(all_song_unfold_df,all_song,how='left',on='level_0')

all_song = []
gc.collect()

all_song_unfold_df = all_song_unfold_df.drop('level_0',axis=1)

print("Information Extraction from the Test Data:")

test_data_path = data_path + 'challenge.v1/'
f = open(test_data_path+'challenge_set.json')
js = f.read()
f.close()
mpd_slice = json.loads(js)
tmp_slice = pd.DataFrame.from_dict(mpd_slice['playlists'])
test_all_song = tmp_slice[['pid','tracks']]
        
test_all_song = test_all_song.reset_index(drop=True)        
    
test_all_song_unfold = test_all_song['tracks'].apply(pd.Series).stack()

test_all_song_unfold_df = pd.DataFrame(test_all_song_unfold)
test_all_song_unfold_df['level_0'] = np.array(test_all_song_unfold.index.get_level_values(0))

cols = ['artist_uri','pos','artist_name','track_uri','track_name','album_uri','duration_ms','album_name']

for col in cols:
    test_all_song_unfold_df[col] = test_all_song_unfold.apply(lambda x:x[col])
    
test_all_song_unfold_df = test_all_song_unfold_df.drop([0],axis=1)

test_all_song_unfold_df = test_all_song_unfold_df.reset_index(drop=True)

gc.collect()

test_all_song_unfold = []

test_all_song['level_0'] = test_all_song.index.get_level_values(0)
test_all_song = test_all_song.drop(['tracks'],axis=1)

test_all_song_unfold_df = pd.merge(test_all_song_unfold_df,test_all_song,how='left',on='level_0')

test_all_song = []
gc.collect()

test_all_song_unfold_df = test_all_song_unfold_df.drop('level_0',axis=1)

n_train = all_song_unfold_df.shape[0]
n_test = test_all_song_unfold_df.shape[0]

train_song_unfold_df = deepcopy(all_song_unfold_df)

all_song_unfold_df = pd.concat((train_song_unfold_df,test_all_song_unfold_df))

gc.collect()

all_song_unfold_df = all_song_unfold_df.reset_index(drop=True)  

print("Numerical Encoding of Album Uri:")
le_album_url = LabelEncoder()
all_song_unfold_df['album_uri'] = le_album_url.fit_transform(all_song_unfold_df['album_uri'])
train_song_unfold_df
joblib.dump(le_album_url, data_path+'le_album_url.pkl')

#le_album_url = []
print("Numerical Encoding of Artist Uri:")
le_artist_uri = LabelEncoder()
all_song_unfold_df['artist_uri'] = le_artist_uri.fit_transform(all_song_unfold_df['artist_uri'])
joblib.dump(le_artist_uri, data_path+'le_artist_uri.pkl')
#le_artist_uri = []
all_song_unfold_df['artist_uri'] = all_song_unfold_df['artist_uri'].astype('int32')

print("Numerical Encoding of Track Uri:")
le_track_uri = LabelEncoder()
all_song_unfold_df['track_uri'] = le_track_uri.fit_transform(all_song_unfold_df['track_uri'])
joblib.dump(le_track_uri, data_path+'le_track_uri.pkl')
#le_track_uri = []
all_song_unfold_df['track_uri'] = all_song_unfold_df['track_uri'].astype('int32')

song_info = all_song_unfold_df.drop_duplicates('track_uri')
song_info = song_info.drop('pos',axis=1)
song_info = song_info.drop('pid',axis=1) 

print("Saving the Song_Info File:")
song_info.to_csv(data_path+'song_info.csv.gz',compression='gzip',index=False)

all_song_unfold_df = []
gc.collect()

train_song_unfold_df = train_song_unfold_df.drop(['artist_uri','artist_name','track_name','album_uri', 'duration_ms', 'album_name'],axis=1)
test_all_song_unfold_df = test_all_song_unfold_df.drop(['artist_uri','artist_name','track_name','album_uri', 'duration_ms', 'album_name'],axis=1)

print("Saving the Train_List_Song_Info File:")
train_song_unfold_df = train_song_unfold_df.reset_index(drop=True)
train_song_unfold_df['track_uri'] = le_track_uri.transform(train_song_unfold_df['track_uri'])
train_song_unfold_df.to_csv(data_path + 'train_list_song_info.csv.gz', compression='gzip', index=False)

print("Saving the Test_List_Song_Info File:")
test_all_song_unfold_df = test_all_song_unfold_df.reset_index(drop=True)
test_all_song_unfold_df['track_uri'] = le_track_uri.transform(test_all_song_unfold_df['track_uri'])
test_all_song_unfold_df.to_csv(data_path + 'test_list_song_info.csv.gz', compression='gzip', index=False)

train_song_unfold_df = []
test_all_song_unfold_df = []
gc.collect()

dirs = os.listdir( data_path + 'mpd.v1/data/' )
count = 0
for json_file in tqdm(dirs):    
#    print(count)
    count = count + 1
    f = open(data_path+'mpd.v1/data/'+json_file)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)
    tmp_slice = pd.DataFrame.from_dict(mpd_slice['playlists'])
    tmp_slice = tmp_slice.drop(['tracks'],axis=1)
    if count == 1:
        all_play_list = tmp_slice
    else:
        all_play_list = pd.concat((all_play_list,tmp_slice))
        
print("Saving the Train_List_Info File:")
        
all_play_list = all_play_list.reset_index(drop=True)
all_play_list.to_csv(data_path+'train_list_info.csv.gz',compression='gzip',index=False)

print("Saving the Test_List_Info File:")

test_data_path = data_path + 'challenge.v1/'
f = open(test_data_path+'challenge_set.json')
js = f.read()
f.close()
mpd_slice = json.loads(js)
tmp_slice = pd.DataFrame.from_dict(mpd_slice['playlists'])
test_all_play_list = tmp_slice.drop(['tracks'],axis=1)
test_all_play_list.to_csv(data_path+'test_list_info.csv.gz',compression='gzip',index=False)
