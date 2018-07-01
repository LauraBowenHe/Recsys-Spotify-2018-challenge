# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:48:40 2018

@author: bwhe
"""

import ast
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import pickle
import w2v
import time
import metric

from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from itertools import repeat


def remove_iteral(sentence):
    return ast.literal_eval(sentence) 

readfile = 'cf2xgb_10.csv.gz'
savefile_train = 'cf2xgb_10_train.csv.gz'
savefile_test = 'cf2xgb_10_test.csv.gz'

nrows = None 
df = pd.read_csv(readfile, usecols=['pid','real', 'pred', 'scores'], nrows=nrows)
df = df.rename(columns={'scores': 'cf_prob'})

df_train, df_test = train_test_split(df, test_size=0.2, random_state=666)
w2v_test_pids = df_test.pid.unique()
# save for later prediction
final_df_test = df_test[['pid', 'real']]
final_df_test['real'] = final_df_test['real'].apply(remove_iteral)

df_train['real'] = df_train['real'].apply(remove_iteral)
df_train['pred'] = df_train['pred'].apply(remove_iteral)
df_train['cf_prob'] = df_train['cf_prob'].apply(remove_iteral)

df_test['real'] = df_test['real'].apply(remove_iteral)
df_test['pred'] = df_test['pred'].apply(remove_iteral)
df_test['cf_prob'] = df_test['cf_prob'].apply(remove_iteral)


with open('result.txt', 'a') as f:
    f.write("original metric values...\n")
    df_test['r_precision'] = df_test.apply(metric.r_precision, axis=1).tolist()
    r_precision_res = np.array(df_test.apply(metric.r_precision, axis=1).tolist())
    print("r_precision is %s"%(np.mean(r_precision_res)))
    f.write("r_precision is %s \n"%(np.mean(r_precision_res)))
    
    df_test['ndcg'] = df_test.apply(metric.ndcg, axis=1).tolist()
    ndcg_res = np.array(df_test.apply(metric.ndcg, axis=1).tolist())
    print("ndcg is %s"%(np.mean(ndcg_res)))
    f.write("ndcg is %s \n"%(np.mean(ndcg_res)))
    
    df_test['pclc'] = df_test.apply(metric.playlist_extender_clicks, axis=1).tolist()
    playclick_res = np.array(df_test.apply(metric.playlist_extender_clicks, axis=1).tolist())
    print("play click is %s"%(np.mean(playclick_res)))
    f.write("play click is %s \n"%(np.mean(playclick_res)))



def real_pred_intersect(row):
    return list(set(row['real'])&set(row['pred']))
start_time = time.time()
df_train['real_inter'] = df_train.apply(real_pred_intersect, axis=1)
df_test['real_inter'] = df_test.apply(real_pred_intersect, axis=1)

print("real pred intersect --- %s seconds ---" % (time.time() - start_time))


''' convert the list mode to column mode '''
start_time = time.time()
result_train = pd.DataFrame([(tup.pid, pred, cf_prob) for tup in df_train.itertuples() 
                        for pred, cf_prob in zip(tup.pred, tup.cf_prob)])
labels_train = pd.DataFrame([(tup.pid, real) for tup in df_train.itertuples()
                            for real in tup.real_inter])

result_test = pd.DataFrame([(tup.pid, pred, cf_prob) for tup in df_test.itertuples() 
                        for pred, cf_prob in zip(tup.pred, tup.cf_prob)])
labels_test = pd.DataFrame([(tup.pid, real) for tup in df_test.itertuples()
                            for real in tup.real_inter])
    
print("convert to column --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
labels_train['label'] = 1
result_train = result_train.merge(labels_train, left_on=[0, 1], right_on=[0,1], how='left')
labels_test['label'] = 1
result_test = result_test.merge(labels_test, left_on=[0, 1], right_on=[0,1], how='left')
print("merge label --- %s seconds ---" % (time.time() - start_time))

result_train = result_train.fillna(0)
result_train.columns = ['pid', 'track_uri', 'cf_prob', 'label']

result_test = result_test.fillna(0)
result_test.columns = ['pid', 'track_uri', 'cf_prob', 'label']
del df_train, df_test, labels_train, labels_test
gc.collect()

''' add relative cf probability score '''
def add_avg_cfscore(data):
    tmp = data.groupby('pid')['cf_prob'].mean().reset_index().rename(columns={'cf_prob':'cf_avg_prob'})
    data = data.merge(tmp, left_on=['pid'], right_on=['pid'], how='left')
    data['cf_rlt_prob'] = data['cf_prob'] / data['cf_avg_prob']
    data = data.drop(['cf_avg_prob'], axis=1)
    return data
start_time = time.time()
result_train = add_avg_cfscore(result_train)
result_test = add_avg_cfscore(result_test)
print("average score --- %s seconds ---" % (time.time() - start_time))

''' add song frequency '''
start_time = time.time()
songfreq = pd.read_csv('../data/songfreq.csv.gz')
result_train = result_train.merge(songfreq, left_on=['track_uri'], right_on=['track_uri'], how='left')
result_test = result_test.merge(songfreq, left_on=['track_uri'], right_on=['track_uri'], how='left')
print("songfreq merge --- %s seconds ---" % (time.time() - start_time))


''' add album uri '''
start_time = time.time()
with open('../data/song2album.pkl', 'rb') as f:
    song2album = pickle.load(f)
tracks_train = result_train['track_uri'].values
album_uri = [song2album[track] for track in tracks_train]
result_train['album_uri'] = album_uri
tracks_test = result_test['track_uri'].values
album_uri = [song2album[track] for track in tracks_test]
result_test['album_uri'] = album_uri
print("add album uri --- %s seconds ---" % (time.time() - start_time))

del album_uri, song2album
gc.collect()

''' add track uri '''
start_time = time.time()
with open('../data/song2artist.pkl', 'rb') as f:
    song2artist = pickle.load(f)
artist_uri_train = [song2artist[track] for track in tracks_train]
result_train['artist_uri'] = artist_uri_train
artist_uri_test = [song2artist[track] for track in tracks_test]
result_test['artist_uri'] = artist_uri_test
print("add artist uri --- %s seconds ---" % (time.time() - start_time))

del artist_uri_train, artist_uri_test, song2artist
gc.collect()

pids_train = result_train['pid']
pids_test = result_test['pid']

''' add similarity between playlist name and track name '''
with open('../data/song2names.pkl', 'rb') as f:
    song2names = pickle.load(f)
song_names_train = [song2names[track] for track in tracks_train]
song_names_test = [song2names[track] for track in tracks_test]
del song2names
gc.collect()


with open('../data/pid2more_clean_name.pkl', 'rb') as f:
    pid2names = pickle.load(f)
pid_names_train = [pid2names[pid] for pid in pids_train]
pid_names_test = [pid2names[pid] for pid in pids_test]
del pid2names
gc.collect()


from difflib import SequenceMatcher
def similar(var):
    a = var[0]
    b = var[1]
    a = str(a).lower()
    b = str(b).lower()
    return SequenceMatcher(None, a, b).ratio()
#name_sim = [similar(str(a).lower(), str(b)) for a, b in zip(song_names, pid_names)]
start_time = time.time()
name_sim_train = list(map(similar, zip(song_names_train, pid_names_train)))
result_train['name_sim'] = name_sim_train
name_sim_test = list(map(similar, zip(song_names_test, pid_names_test)))
result_test['name_sim'] = name_sim_test
print("calculate track name similarity --- %s seconds ---" % (time.time() - start_time))


''' add similarity between playlist name and album name '''
with open('../data/song2album_name.pkl', 'rb') as f:
    song2album_names = pickle.load(f)
album_names_train = [song2album_names[track] for track in tracks_train]
album_names_test = [song2album_names[track] for track in tracks_test]

start_time = time.time()
album_sim_train = list(map(similar, zip(album_names_train, pid_names_train)))
result_train['album_sim'] = album_sim_train
album_sim_test = list(map(similar, zip(album_names_test, pid_names_test)))
result_test['album_sim'] = album_sim_test
print("calculate album similarity --- %s seconds ---" % (time.time() - start_time))

del song2album_names, album_names_train, album_names_test
gc.collect()

''' add similarity between playlist name and artist name '''
start_time = time.time()
with open('../data/song2artist_name.pkl', 'rb') as f:
    song2artist_names = pickle.load(f)
artist_names_train = [song2artist_names[track] for track in tracks_train]
artist_sim_train = list(map(similar, zip(artist_names_train, pid_names_train)))
result_train['artist_sim'] = artist_sim_train

artist_names_test = [song2artist_names[track] for track in tracks_test]
artist_sim_test = list(map(similar, zip(artist_names_test, pid_names_test)))
result_test['artist_sim'] = artist_sim_test
print("calculate artist name similarity --- %s seconds ---" % (time.time() - start_time))
del song2artist_names, artist_names_train, artist_names_test
gc.collect()

''' add similarity '''
#result.to_csv(savefile, index=False, compression='gzip')
w2v.build_track_w2v(w2v_test_pids)
w2v.build_album_w2v(w2v_test_pids)
w2v.build_artist_w2v(w2v_test_pids)

from gensim.models import Word2Vec
model1 = Word2Vec.load('w2v_model1.bin')
model2 = Word2Vec.load('w2v_model2.bin')
model3 = Word2Vec.load('w2v_model3.bin')


with open('../data/song2album.pkl', 'rb') as f:
        song2album = pickle.load(f)

with open('../data/song2artist.pkl', 'rb') as f:
        song2artist = pickle.load(f)

def remove_iteral(sentence):
    return ast.literal_eval(sentence) 


df = pd.read_csv(readfile, usecols=['pid','pos_songs'], nrows=None)
df['pos_songs'] = df['pos_songs'].apply(remove_iteral)

result_train = result_train.merge(df, left_on=['pid'], right_on=['pid'], how='left')
result_test = result_test.merge(df, left_on=['pid'], right_on=['pid'], how='left')

def track_sim(var):
    pos_song = var[0]
    track = var[1]
    try:
        return model1.wv.similarity(str(pos_song), str(track))
    except:
        return 0

def album_sim(var):
    pos_song = var[0]
    track = var[1]
    try:
        return model2.wv.similarity(str(song2album[pos_song]), str(song2album[track]))
    except:
        return 0

def artist_sim(var):
    pos_song = var[0]
    track = var[1]
    try:
        return model3.wv.similarity(str(song2artist[pos_song]), str(song2artist[track]))
    except:
        return 0

def w2v_sim(tup):
    pos_songs = tup.pos_songs 
    track = tup.track_uri
    track_scores = list(map(track_sim, zip(pos_songs, repeat(track))))
    album_scores = list(map(album_sim, zip(pos_songs, repeat(track))))
    artist_scores = list(map(artist_sim, zip(pos_songs, repeat(track))))
    return np.mean(track_scores), np.mean(album_scores), np.mean(artist_scores)

def add_w2v_sim(data):
    track_w2v_sim_arr = []
    album_w2v_sim_arr = []
    artist_w2v_sim_arr = []
    for tup in data.itertuples():
        track_score, album_score, artist_score = w2v_sim(tup)
        track_w2v_sim_arr.append(track_score)
        album_w2v_sim_arr.append(album_score)
        artist_w2v_sim_arr.append(artist_score)
    data['w2v_track_sim'] = track_w2v_sim_arr
    data['w2v_album_sim'] = album_w2v_sim_arr
    data['w2v_artist_sim'] = artist_w2v_sim_arr
    return data

start_time = time.time()
print("Calculate word2vec similarity, this takes a long time, like 1.5h.\n Please be patient...")
result_train = add_w2v_sim(result_train)
result_test = add_w2v_sim(result_test)
print("track w2v similarity --- %s seconds ---" % (time.time() - start_time))



result_train.to_csv(savefile_train, index=False, compression='gzip')
result_test.to_csv(savefile_test, index=False, compression='gzip')

