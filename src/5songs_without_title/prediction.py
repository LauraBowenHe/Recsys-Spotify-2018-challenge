# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:02:56 2018

@author: bwhe
"""


import ast
import os
import math
import sys
import scipy.sparse as sp
import pickle
import MSD_rec, util
import numpy as np
import pandas as pd



############################################################################
### generate train and test
############################################################################
fname = '../data/train_list_song_info.csv.gz'
savepath = '../submit/'
nrows = None

version = sys.argv[1]
threshould = sys.argv[2]

util.train_test4pred(fname, version, nrows, threshould)
util.song2user_songcount(threshould) 
util.song2usercount(threshould)
 

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
mat = sp.dok_matrix((2262292, 1000000), dtype=np.int64)


for song, pids in dic.items():
    for pid in pids:
        mat[song, pid] = 1
mat = mat.tocsr().sorted_indices()
coomat = mat.tocoo()

_A = 0.1
_Q = 0.5



#############################################################################
### build predict class
#############################################################################
pr=MSD_rec.PredSI(s2u_tr, _A, _Q, mat, coomat, s2uc)

#############################################################################
### build recommender class
#############################################################################
print('Creating recommender..')
cp = MSD_rec.SReco(songs_ordered, _A, threshould, s2uc)
cp.Add(pr)
cp.Gamma=[1.0]

r = cp.RecommendToUsers(test_users_v, u2s_v, test_u2s_v)
r = r.astype(np.int64)


print("load test...")
test_fname = 'test_pid2songs_'+str(threshould)+'.csv.gz'
test = pd.read_csv(test_fname, usecols=['pid'])


pred = np.zeros([test.shape[0], 500+1])
for i in range(pred.shape[0]):
    pred[i][0] = test['pid'][i]
    pred[i][1:] = r[i]

filename = 'cf2submit_v'+str(version)+'.txt'
np.savetxt(os.path.join(savepath, filename), pred, delimiter=',')
