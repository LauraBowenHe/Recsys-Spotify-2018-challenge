import gc
import pickle
import pandas as pd


def train_test4pred(listfname, songfname, nrows, threshould):
    df_train = pd.read_csv(listfname, usecols=['pid', 'more_clean_name'], nrows=nrows)
    df_train = df_train.dropna().reset_index()
    songfile = pd.read_csv(songfname)
    df = df_train.merge(songfile, left_on=['pid'], right_on=['pid'], how='left')
    print("train song shape is %s"%(df.shape[0]))
    all_pid = df.pid.unique()
    list_song = df.groupby('pid').apply(lambda x: list(x.track_uri))
    train = {}
    pids = []
    for pid in all_pid:
        res = list_song[pid]
        pids.append(pid)
        train[pid] = res
    del df, list_song
    gc.collect()
     
                       
    train_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
    with open(train_fname, 'wb') as f:
        pickle.dump(train, f)
    
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
