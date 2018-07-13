# -*- coding: utf-8 -*-
"""
use nlp and cf
Created on Fri Jun 15 22:56:49 2018

@author: bwhe
"""


import gc
import pickle
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import MSD_rec, util


class BuildMat(object):
    def __init__(self, df, all_txt, pids):
        self.df = df
        self.txt = all_txt
        self.pids = pids
        self.dic = {}
        self.total_songs = 0

    def build_all_train(self):
        count_vec = CountVectorizer(analyzer='word', 
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)
        count_train = count_vec.fit(self.txt)
        return count_train
    
    def build_word_index(self, count_train, fname):
        i = 0
        for vocab, _ in count_train.vocabulary_.items():
            (self.dic)[vocab] = i
            i += 1
        with open(fname, 'wb') as f:
            pickle.dump((self.dic), f)
     
    def build_spmat(self, count_train):
        train = pd.read_csv('../data/train_list_song_info.csv.gz', usecols=['pid', 'track_uri'])
        train = train[train['pid'].isin(self.pids)].reset_index()
        print(train.shape)
        shape = len(count_train.vocabulary_)
        mat = sp.dok_matrix((shape, 2262292), dtype=np.int64)
        with open('../data/pid2more_clean_name.pkl', 'rb') as f:
            pid2name = pickle.load(f)
        for i in range(train.shape[0]):
            word_vec = CountVectorizer(analyzer='word', 
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)
            try:
                column = train['track_uri'][i]
                name = pid2name[train['pid'][i]]
            except:
                continue
#                print("wrong for column...{}".format(i))
            try:
                words = word_vec.fit([name])
            except:
#                print("wrong words fit ...{}".format(name))
                continue
            for item in words.get_feature_names():
                try:
                    row = (self.dic)[item]
                except:
#                    print("wrong item...{}".format(item))
                    continue
                try:
                    if len(name.split(' ')) == 1:
                        mat[row, column] += 1
                    else:
                        mat[row, column] += 3
                except:
                    if len(name.split(' ')) == 1:
                        mat[row, column] = 1
                    else:
                        mat[row, column] = 3
        mat = mat.tocsr().sorted_indices()
        print("finish build sparse matrix...")
        return mat
    

class Pred(object):
    def __init__(self, df):
        self.df = df
        self.word_vec = CountVectorizer(analyzer='word', 
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)
        self.incomplete_name = 0
        self.missing_words = 0

        
    def rec4list(self, word_index, name, mat, cp_song):
        try:
            words = (self.word_vec).fit([name])
        except:
#            print("-----------no word, random-------------")
            return np.random.choice(np.asarray(range(2262292)), 500, replace=False)
        column_list = []
        for item in words.get_feature_names():
            try:
                column_list.append(word_index[item])
            except:
                self.missing_words += 1
                
        """ if all words do not appear before, use random to help """
        if len(column_list) == 0:
#            print("-----not appear before, random-----")
            return np.random.choice(np.asarray(range(2262292)), 500, replace=False)
               
        row0 = mat.getrow(column_list[0])
        if len(column_list) > 1:
            for column in column_list[1:]:
                row1 = mat.getrow(column)
                row0 += row1
        res = np.asarray(row0.todense())[0]
        
        ''' use nearest neighbour and cf to second time predict '''
        ssongs = np.argsort(np.asarray(res))[-100:][::-1]
        scores = [res[song] for song in ssongs]
        res_neighbour = cp_song.RecomToTitle(ssongs, scores)
        fst_weight = 0.5
        snd_weight = 0.5
        res2 = res * fst_weight + res_neighbour * snd_weight
        ssongs2 = np.argsort(np.asarray(res2))[-500:][::-1].tolist()
        return ssongs2
        
    def rec4lists(self, mats, cp_song):
        num_list = (self.df).shape[0]
        res = np.zeros((num_list, 501))
        
        with open('word_index.pkl', 'rb') as f:
            word_index = pickle.load(f)
        
        with open('../data/test_pid2more_clean_name.pkl', 'rb') as f:
            pid2name = pickle.load(f)
        
        for i in range(num_list):
            name = pid2name[(self.df)['pid'][i]]
            ''' case for title only '''
            tmp = self.rec4list(word_index, name, mat, cp_song)
            res[i, 0] = (self.df)['pid'][i]
            res[i, 1:] = tmp
            if i % 10 == 0:
                print("recommend %s playlists"%(i))
        return res.astype(np.int32)
    
 
''' build train data '''
nrows = None
df = pd.read_csv('../data/train_list_info.csv.gz', usecols=['pid', 'more_clean_name'], nrows=nrows)
df = df.dropna()
pids = list(df['pid'])
txt = list(df['more_clean_name'])


print("build sparse matrix...")
buildmat = BuildMat(df, txt, pids)
del df, pids, txt
gc.collect()

count_train = buildmat.build_all_train()
buildmat.build_word_index(count_train, 'word_index.pkl')
mat = buildmat.build_spmat(count_train)

''' build test data '''
with open('../data/pred_v1.txt', 'rb') as f:
    pids = pickle.load(f)
df = pd.DataFrame({'pid':pids})
print("build predictor...")
pr = Pred(df)

###########################################################################################
########################         nearest neighbour cf       ###############################
###########################################################################################
threshould = 0
version = 1
nrows = None
listfname = "../data/train_list_info.csv.gz"
songfname = "../data/train_list_song_info.csv.gz"
util.train_test4pred(listfname,  songfname, nrows, threshould)
util.song2user_songcount(threshould) 
util.song2usercount(threshould)
 

print('default ordering by popularity')
train_songs_ordered_fname = 'train_songs_ordered_'+str(threshould)+'.pkl'
with open(train_songs_ordered_fname, 'rb') as f:
    songs_ordered = pickle.load(f)

print('song to pids on train_song2pids_10.pkl')
train_song2pid_fname = 'train_song2pids_'+str(threshould)+'.pkl'
with open(train_song2pid_fname, 'rb') as f:
    s2u_tr = pickle.load(f)


print('user to songs on train_user_to_song')
train_pid2songs_fname = 'train_pid2songs_'+str(threshould)+'.pkl'
with open(train_pid2songs_fname, 'rb') as f:
    u2s_v = pickle.load(f)
    
print("song to pid count on train")
s2uc_fname = 'train_songs_pidcount_'+str(threshould)+'.pkl'
with open(s2uc_fname, 'rb') as handle:
    s2uc = pickle.load(handle)

print('Creating predictor..')
train_song2pids_fname = 'train_song2pids_'+str(threshould)+'.pkl'
with open(train_song2pids_fname, 'rb') as f:
    dic = pickle.load(f)
mat_song = sp.dok_matrix((2262292, 1000000), dtype=np.int64)

for song, pids in dic.items():
    for pid in pids:
        mat_song[song, pid] = 1
mat_song = mat_song.tocsr().sorted_indices()
coomat_song = mat_song.tocoo()

_A = 0.1
_Q = 0.2


pr_song = MSD_rec.PredSI(s2u_tr, _A, _Q, mat_song, coomat_song, s2uc)

#############################################################################
### build recommender class
#############################################################################
print('Creating recommender..')
cp_song = MSD_rec.SReco(songs_ordered, _A, threshould, s2uc, u2s_v)
cp_song.Add(pr_song)
cp_song.Gamma=[1.0]

print("start to recommend...")

res = pr.rec4lists(mat, cp_song)

gc.collect()
np.savetxt('../submit/tosubmit_v1.csv', res, delimiter=',')
