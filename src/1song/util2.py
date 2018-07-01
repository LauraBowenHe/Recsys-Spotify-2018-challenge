# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:35:39 2018

@author: bwhe
"""


import gc
import pickle
import pandas as pd

def get_train_test_1(train_listfname, test_listfname, songfname, nrows, testrows, threshould=5):
    songfile = pd.read_csv(songfname)
    test = {}
    test_pos_songs = []
    test_songs = []
    df_test = pd.read_csv(test_listfname, usecols = ['pid', 'more_clean_name'], nrows=testrows)
    df_test = df_test.dropna().reset_index()
    df = df_test.merge(songfile, left_on=['pid'], right_on=['pid'], how='left')
    
    print("test song shape is %s"%(df.shape[0]))
    test_all_pid = df_test['pid'].tolist()
    list_song = df.groupby('pid').apply(lambda x: list(x.track_uri))
    for pid in test_all_pid:
        res = list_song[pid]
        test[pid] = res[0:threshould]
        test_pos_songs.append(res[0:threshould])
        test_songs.append(res[threshould:])
    
    df_train = pd.read_csv(train_listfname, usecols=['pid', 'more_clean_name', 'modified_at'], nrows=nrows)
    df_train = df_train.dropna().reset_index()
    print(df_train.shape)
    df_train = df_train[~df_train['pid'].isin(test_all_pid)]
    print(df_train.shape)
    df = df_train.merge(songfile, left_on=['pid'], right_on=['pid'], how='left')
    print("train song shape is %s"%(df.shape[0]))
    all_pid = df_train.pid.unique()
    list_song = df.groupby('pid').apply(lambda x: list(x.track_uri))
    train = {}
    pids = []
    for pid in all_pid:
        if pid in test_all_pid:
            continue
        res = list_song[pid]
        pids.append(pid)
        train[pid] = res
    del df, list_song
    gc.collect()
       
    test_fname = 'test_pid2songs_'+str(threshould)+'.pkl'
    with open(test_fname, 'wb') as f:
        pickle.dump(test, f)
    
    train_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
    with open(train_fname, 'wb') as f:
        pickle.dump(train, f)
    
    pid_fname = 'only_pid_list_'+str(threshould)+'.pkl'
    with open(pid_fname, 'wb') as f:
        pickle.dump(pids, f)
    
    pid_fname = 'test_pid_'+str(threshould)+'.pkl' 
    with open(pid_fname, 'wb') as f:
        pickle.dump(test_all_pid, f)
    
    test_fname = 'test_pid2songs_'+str(threshould)+'.csv.gz'
    test = pd.DataFrame({
            'pid':test_all_pid,
            'pos_songs': test_pos_songs,
            'real':test_songs
            })
    test.to_csv(test_fname, compression='gzip')
    return df_train, df_test

        
def sort_dict_dec(d):
    return sorted(d.keys(),key=lambda s:d[s],reverse=True)

def song2usercount(threshould):
    s2uc = {}
    with open('train_song2pids_'+str(threshould)+'.pkl', 'rb') as f:
        dic = pickle.load(f)
        for song, pids in dic.items():
            s2uc[song] = len(pids)
    test_s2u = {}
    with open('test_pid2songs_'+str(threshould)+'.pkl', 'rb') as f:
        test =pickle.load(f)
        for pid, songs in test.items():
            for song in songs:
                if song not in test_s2u:
                    test_s2u[song] = set([pid])
                else:
                    test_s2u[song].add(pid)
    for song in range(2262292):
        if song not in s2uc:
            s2uc[song] = 0
        
    with open('train_songs_pidcount_'+str(threshould)+'.pkl', 'wb') as f:
        pickle.dump(s2uc, f)
        
def song2user_songcount(threshould):
    songs_ordered = {}
    s2u = {}
    with open('train_pid2songs_'+str(threshould)+'.pkl', 'rb') as f:
        train = pickle.load(f)
        for pid, songs in train.items():
            for song in songs:
                if song not in songs_ordered:
                    songs_ordered[song] = 1
                else:
                    songs_ordered[song] += 1
                
                if song not in s2u:
                    s2u[song] = set([pid])
                else:
                    s2u[song].add(pid)
    
    songs_ordered = sort_dict_dec(songs_ordered)
    with open('train_songs_ordered_'+str(threshould)+'.pkl', 'wb') as fin:
        pickle.dump(songs_ordered, fin)
    
    with open('train_song2pids_'+str(threshould)+'.pkl', 'wb') as f:
        pickle.dump(s2u, f)