# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:39:59 2018

@author: bwhe
"""

import pickle
import numpy as np
import pandas as pd
import gc

#########################################################################################
###   get specific playlist  id for 10 sub tasks                    
#########################################################################################
df = pd.read_csv('../data/test_list_info.csv.gz')

print("only title pids...")
pid1 = df[df['num_samples']==0].pid.unique()
with open('../data/pred_v1.txt', 'wb') as fp:
    pickle.dump(pid1, fp)

print("1 song with title samples pids...")
pid2 = df[df['num_samples']==1].pid.unique()
with open('../data/pred_v2.txt', 'wb') as fp:
    pickle.dump(pid2, fp)
    
print("5 songs with title samples pids...")
pid3 = df[(df['num_samples']==5)&(~df['name'].isnull())].pid.unique()
with open('../data/pred_v3.txt', 'wb') as fp:
    pickle.dump(pid3, fp)    
    
print("5 songs without title samples pids...")
pid4 = df[(df['num_samples']==5)&(df['name'].isnull())].pid.unique()
with open('../data/pred_v4.txt', 'wb') as fp:
    pickle.dump(pid4, fp)    

print("10 with title samples pids...")
pid5 = df[(df['num_samples']==10)&(~df['name'].isnull())].pid.unique()
with open('../data/pred_v5.txt', 'wb') as fp:
    pickle.dump(pid5, fp)
    
print("10 without title samples pids...")
pid6 = df[(df['num_samples']==10)&(df['name'].isnull())].pid.unique()
with open('../data/pred_v6.txt', 'wb') as fp:
    pickle.dump(pid6, fp)  

print("25 songs samples pids...")
pid7 = df[df['num_samples']==25].pid.unique()
with open('../data/pred_v7.txt', 'wb') as fp:
    pickle.dump(pid7, fp)

    
print("100 songs samples pids...")
pid8 = df[df['num_samples']==100].pid.unique()
with open('../data/pred_v8.txt', 'wb') as fp:
    pickle.dump(pid8, fp)  


# extract sampels in order and samples by shuffle for sampels length is 25 and 100
print("extract 25 songs and 100 songs in order and shuffle...")
test_songs = pd.read_csv('../data/test_list_song_info.csv.gz')

def check_order(group):
    if group.is_monotonic and np.sum(np.diff(group)) == _SUM_:
        return 1
    else:
        return 0

_SUM_ = 24
test_songs_25 = test_songs[test_songs['pid'].isin(pid7)]
tmp = test_songs_25.groupby('pid')['pos'].apply(check_order).reset_index()

sample_25_order_pids = np.asarray(tmp[tmp['pos']==1].pid)
sample_25_shuffle_pids = np.asarray(tmp[tmp['pos']==0].pid)
np.savetxt('../data/samples_25_order_pids.txt.gz', sample_25_order_pids, delimiter=',')
np.savetxt('../data/samples_25_shuffle_pids.txt.gz', sample_25_shuffle_pids, delimiter=',')
del test_songs_25, tmp
gc.collect()

test_songs_100 = test_songs[test_songs['pid'].isin(pid8)]
_SUM_ = 99
tmp = test_songs_100.groupby('pid')['pos'].apply(check_order).reset_index()

sample_100_order_pids = np.asarray(tmp[tmp['pos']==1].pid)
sample_100_shuffle_pids = np.asarray(tmp[tmp['pos']==0].pid)
np.savetxt('../data/samples_100_order_pids.txt.gz', sample_100_order_pids, delimiter=',')
np.savetxt('../data/samples_100_shuffle_pids.txt.gz', sample_100_shuffle_pids, delimiter=',')
del test_songs_100, tmp
gc.collect()


################################################################################################
### ensemble playlist pids and provided samples 
################################################################################################
print("merge playlist pids and provied tracks...")
def build_test_list(version):
    version = version + 2
    version = str(version)
    readfile = '../data/pred_v'+version+'.txt'
    savefile = '../data/test_list_song_v'+version+'.csv.gz'
    with open(readfile, 'rb') as fp:
        pid_list = pickle.load(fp)
    
    pids = []
    songs = []
    for pid in pid_list:
        songs.append(list_song[pid])
        pids.append(pid)

        
    pid_song = pd.DataFrame({
                'pid': pids,
                'songs': songs
            })
    
    pid_song.to_csv(savefile, compression='gzip')

list_song = test_songs.groupby('pid').apply(lambda x:list(x.track_uri))
for i in range(7):
    build_test_list(i)
    

################################################################################################
### extract playlist in train dataset(MSD) whose length is within the range with test samples
################################################################################################
print("extract palylist whose length is compatible with same length...")
train_list_info = pd.read_csv('../data/train_list_info.csv.gz')
def pid_within_length(df, min_len, max_len, given_num):
    df = df[(df['num_tracks'] <= max_len) & (df['num_tracks'] >= min_len)]
    savefile = '../data/train_list_info_title'+str(given_num)+'songs.csv.gz'
    df.to_csv(savefile, index=False, compression='gzip')
    
# for 1 song task
pid_within_length(train_list_info, 11, 80, 1)

# for 5 and 10 song task
pid_within_length(train_list_info, 40, 100, 5)
# for 10 song task
pid_within_length(train_list_info, 40, 100, 10)
