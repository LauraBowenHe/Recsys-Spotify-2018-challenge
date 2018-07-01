# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:31:49 2018

@author: bwhe
"""


import ast
import scipy.sparse as sp
import pickle
import MSD_rec2, util2, metric
import numpy as np
import pandas as pd



############################################################################
### generate train and test
############################################################################
train_listfname = '../data/train_list_info.csv.gz'
test_listfname = '../data/train_list_info_title1songs.csv.gz'
songfname = '../data/train_list_song_info.csv.gz'

nrows = 990000
testrows = 12000
threshould = 1


util2.get_train_test_1(train_listfname, test_listfname, songfname, nrows, testrows, threshould)
util2.song2user_songcount(threshould) 
util2.song2usercount(threshould)
 

print('loading pids...')
test_user_fname = 'test_pid_'+str(threshould)+'.pkl'
with open(test_user_fname, 'rb') as f:
    test_users_v = pickle.load(f)
    print("the first pid in test is ---------")
    print(test_users_v[0])


print('default ordering by popularity')
train_songs_ordered_fname = 'train_songs_ordered_'+str(threshould)+'.pkl'
with open(train_songs_ordered_fname, 'rb') as f:
    songs_ordered = pickle.load(f)

print('song to pids on train_song2pids_10.pkl')
train_song2pid_fname = 'train_song2pids_'+str(threshould)+'.pkl'
with open(train_song2pid_fname, 'rb') as f:
    s2u_tr = pickle.load(f)
print(type(s2u_tr))

print('pid to song on test song2pid...')
test_pid2songs_fname = 'test_pid2songs_'+str(threshould)+'.pkl'
with open(test_pid2songs_fname, 'rb') as f:
    test_u2s_v = pickle.load(f)

print('user to songs on train_user_to_song')
train_pid2songs_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
with open(train_pid2songs_fname, 'rb') as f:
    u2s_v = pickle.load(f)
    
print("song to pid count on train")
s2uc_fname = 'train_songs_pidcount_'+str(threshould)+'.pkl'
with open(s2uc_fname, 'rb') as handle:
    s2uc = pickle.load(handle)

print('Creating predictor..')
############################################################################
### build the user-item sparse matrix
############################################################################
train_song2pids_fname = 'train_song2pids_'+str(threshould)+'.pkl'
with open(train_song2pids_fname, 'rb') as f:
    dic = pickle.load(f)
mat = sp.dok_matrix((2262292, 1000000), dtype=np.float32)


for song, pids in dic.items():
    for pid in pids:
        mat[song, pid] = 1
mat = mat.tocsr().sorted_indices()
coomat = mat.tocoo()

_A = 0.1
_Q = 0.5


with open('result.txt', 'a') as f:
    f.write('threshould = %s nrows = %s testrows = %s \n'%(threshould, nrows, testrows))

#############################################################################
### build predict class
#############################################################################
pr=MSD_rec2.PredSI(s2u_tr, _A, _Q, mat, coomat, s2uc)

#############################################################################
### build recommender class
#############################################################################
print('Creating recommender..')
cp = MSD_rec2.SReco(songs_ordered, _A, threshould, s2uc, u2s_v)
cp.Add(pr)
cp.Gamma=[1.0]

r1, scores = cp.RecommendToUsers(test_users_v, u2s_v, test_u2s_v)
r1 = r1.astype(np.int64)

import gc
del mat, pr, cp
gc.collect()

print("The number of list %s"%(len(r1)))
print("the number of each list %s"%(len(r1[0])))

print("load test...")
test_fname = 'test_pid2songs_'+str(threshould)+'.csv.gz'
test = pd.read_csv(test_fname)
def remove_iteral(sentence):
    return ast.literal_eval(sentence)  
test['real'] = test['real'].apply(remove_iteral)

print("the test shape is %s"%(test.shape[0]))
print("--------the first pid in test is %s------------"%(test['pid'][0]))
''' cf '''
test['pred'] = r1.tolist()
test['scores'] = scores.tolist()
print("evaluation metric...")
r_precision_res = np.array(test.apply(metric.r_precision, axis=1).tolist())
print("r_precision is %s"%(np.mean(r_precision_res)))
ndcg_res = np.array(test.apply(metric.ndcg, axis=1).tolist())
print("ndcg is %s"%(np.mean(ndcg_res)))
playclick_res = np.array(test.apply(metric.playlist_extender_clicks, axis=1).tolist())
print("play click is %s"%(np.mean(playclick_res)))



with open("result.txt", 'a') as f:
    f.write("r_precision is %s \n"%(np.mean(r_precision_res)))
    f.write("ndcg is %s \n"%(np.mean(ndcg_res)))
    f.write("play click is %s \n"%(np.mean(playclick_res)))
    f.write("--------------------------\n")
test.to_csv('cf2xgb_1.csv.gz', index=False, compression='gzip')
