# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:18:22 2018

@author: bwhe
"""

import ast
import gc
import pickle
import numpy as np
import pandas as pd


def train_test4pred(fname, version, nrows=None, threshould=10):
    df = pd.read_csv(fname, nrows=nrows)
    all_pid = df.pid.unique()
    list_song = df.groupby('pid').apply(lambda x: list(x.track_uri))
    train = {}
    pids = []
    
    test_pids = []
    test_pos_songs = []
    
    for pid in all_pid:
        pids.append(pid)
        train[pid] = list_song[pid]
    
    test_dic = {}
    test_fname = '../data/test_list_song_v'+str(version)+'.csv.gz'
    test = pd.read_csv(test_fname)
    
    for i in range(test.shape[0]):
        pid = int(test['pid'][i])
        '''
        pids.append(pid)
        '''
        test_pids.append(pid)
        pos_songs = ast.literal_eval(test['songs'][i])
        '''
        train[pid] = pos_songs
        '''
        test_dic[pid] = list(pos_songs)
        test_pos_songs.append(pos_songs)       
         
    test_fname = 'test_pid2songs_'+str(threshould)+'.pkl'
    with open(test_fname, 'wb') as f:
        pickle.dump(test_dic, f)

    train_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
    with open(train_fname, 'wb') as f:
        pickle.dump(train, f)
    
    pid_fname = 'test_pid_'+str(threshould)+'.pkl' 
    with open(pid_fname, 'wb') as f:
        pickle.dump(test_pids, f)
    
    test_fname = 'test_pid2songs_'+str(threshould)+'.csv.gz'
    test = pd.DataFrame({
            'pid':test_pids,
            'pos_songs': test_pos_songs,
            })
    test.to_csv(test_fname, compression='gzip')
    
    pid_fname = 'only_pid_list_'+str(threshould)+'.pkl'
    with open(pid_fname, 'wb') as f:
        pickle.dump(pids, f)

def train_test_split(fname, nrows=None, threshould=10):
    df = pd.read_csv(fname, skiprows=200001, header=None, names=['pos', 'track_uri', 'pid'], nrows=nrows)
    all_pid = df.pid.unique()
    list_song = df.groupby('pid').apply(lambda x: list(x.track_uri))
    train = {}
    test_pids = []
    test_pos_songs = []
    test_songs = []
    pids = []
    
    del df
    gc.collect()
    
    for pid in all_pid:
        res = list_song[pid]
        if len(res) <= threshould:
            pids.append(pid)
            train[pid] = res
        else:
            # we only want 1000 list for test
            '''
            if len(test_pids) < 100:
                test_pids.append(pid)
                test_pos_songs.append(res[0:threshould])
                test_songs.append(res[threshould:])
            pids.append(pid)
            train[pid] = res[0:threshould]
            '''
            if len(test_pids) < 100:
                test_pids.append(pid)
                pos_songs = list(np.random.choice(res, size=threshould, replace=False))
                rest_songs = list(np.setdiff1d(res, pos_songs))
                test_pos_songs.append(pos_songs)
                test_songs.append(rest_songs)
            pids.append(pid)
            train[pid] = pos_songs
    
    train_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
    with open(train_fname, 'wb') as f:
        pickle.dump(train, f)
    
    pid_fname = 'test_pid_'+str(threshould)+'.pkl' 
    with open(pid_fname, 'wb') as f:
        pickle.dump(test_pids, f)
    
    test_fname = 'test_pid2songs_'+str(threshould)+'.csv.gz'
    test = pd.DataFrame({
            'pid':test_pids,
            'pos_songs': test_pos_songs,
            'real':test_songs
            })
    test.to_csv(test_fname, compression='gzip')
    
    pid_fname = 'only_pid_list_'+str(threshould)+'.pkl'
    with open(pid_fname, 'wb') as f:
        pickle.dump(pids, f)
        


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
    for song, pid in test_s2u.items():
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
        

#train_test_split('../train_list_song_info.csv.gz', 200000, 10)
#song2user_songcount(10)  
          
