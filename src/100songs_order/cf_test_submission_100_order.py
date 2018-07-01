# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 03:13:05 2018
@author: SC
"""

from numpy import *
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from scipy.sparse import coo_matrix, lil_matrix, hstack, vstack, csr_matrix
import pandas as pd
from multiprocessing import Pool
from time import time
from sklearn.svm import LinearSVC
import numpy as np

def cf_predict_evaluation_slim_list(index):
    
    Cr = 2e-1
    beta = 0.7
    theta = 0.99
    alpha = 0.9
    
    song_index_i = test_list_song_info_array[test_list_song_info_array[:, 0] == index, 1]
    song_index_i = array(song_index_i)
    song_index_i = song_index_i.flatten()

    weight = train_list_song_co_occur[song_index_i, :].multiply(power(1e-1 + I_list, beta - 1)).sum(axis=0)
    weight = array(weight).flatten()
    weight = power(weight,theta)
    value = train_list_song_co_occur[song_index_i, :].multiply(weight)
    value = value.dot(train_list_song_co_occur.transpose()) 
    I_song_i = power(1e-1+I_song[song_index_i],-alpha)
    value = value.multiply(I_song_i.reshape((-1,1)))
    value = value.multiply(power(1e-1+I_song,alpha-1))
    value = csr_matrix(value)

    predictions = lil_matrix(value)
    label = zeros(train_list_song_co_occur.shape[0])
    label[song_index_i] = 1
    
    clf = LinearSVC(C=Cr,class_weight={0:1,1:1},tol=1e-6)

    clf.fit(predictions.transpose(),label)
    predictions = clf.decision_function(predictions.transpose())

    predictions = array(predictions).flatten()
    predictions = predictions - min(predictions)
    predictions[song_index_i] = -1e20
    predictions = argsort(predictions)[::-1][:max_n_predictions]

    return predictions


def cf_predict_evaluation_slim(index):
    
    song_index_i = test_list_song_info_array[test_list_song_info_array[:, 0] == index, 1]
    song_index_i = array(song_index_i)
    song_index_i = song_index_i.flatten()

    value = train_list_song_co_occur[song_index_i,:].dot(train_list_song_co_occur.transpose())
    I_song_i = power(1e-1+I_song[song_index_i],-alpha)
    value = value.multiply(I_song_i.reshape((-1,1)))
    value = value.multiply(power(1e-1+I_song,alpha-1))
    value = csr_matrix(value)

    predictions = value
    label = zeros(train_list_song_co_occur.shape[0])
    label[song_index_i] = 1

    clf = LinearSVC(C=5e-2,class_weight={0:1,1:1})

    clf.fit(predictions.transpose(),label)
    predictions = clf.decision_function(predictions.transpose())

    predictions = array(predictions).flatten()
    predictions[song_index_i] = -1e20
    predictions = argsort(predictions)[::-1][:max_n_predictions]

    return predictions

max_n_predictions = 500

data_path = '../data/'

train_list_song_info = pd.read_csv(data_path+'train_list_song_info.csv.gz')
#['pos', 'track_uri', 'pid']

song_info = pd.read_csv(data_path+'song_info.csv.gz')
#['artist_uri', 'artist_name', 'track_uri', 'track_name', 'album_uri',
#       'duration_ms', 'album_name']
train_list_info = pd.read_csv(data_path+'train_list_info.csv.gz')
#['collaborative', 'description', 'duration_ms', 'modified_at', 'name',
#       'num_albums', 'num_artists', 'num_edits', 'num_followers', 'num_tracks',
#       'pid']
test_list_info = pd.read_csv(data_path+'test_list_info.csv.gz')
#['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']
test_list_song_info = pd.read_csv(data_path+'test_list_song_info.csv.gz')
#['pos', 'track_uri', 'pid']
le_track_uri = joblib.load(data_path+'le_track_uri.pkl')

le = LabelEncoder()
train_list_info['pid'] = le.fit_transform(train_list_info['pid'])
train_list_song_info['pid'] = le.transform(train_list_song_info['pid'])

n_songs = max(train_list_song_info['track_uri']) + 1
n_train_lists = train_list_info.shape[0]

print("Constructing song-list co-occurrence matrix:")
train_list_song_co_occur = coo_matrix((ones(train_list_song_info.shape[0]),(array(train_list_song_info['pid']),array(train_list_song_info['track_uri']))),shape=(n_train_lists,n_songs))

gc.collect()

train_list_song_co_occur = train_list_song_co_occur.transpose()

train_list_song_co_occur = lil_matrix(train_list_song_co_occur)

test_list_song_info_array = array(test_list_song_info[['pid','track_uri']])
train_list_song_info_array = array(train_list_song_info[['pid','track_uri']])

pid_array = loadtxt(data_path + 'samples_100_order_pids.txt.gz')
pid_array = pid_array.flatten()
pid_array = pid_array.astype(np.int64)

test_list_pid = pid_array*1

gc.collect()                                                                                                                                                                                                                 

I_song = array(train_list_song_co_occur.sum(axis=1)).flatten()
I_list = array(train_list_song_co_occur.sum(axis=0)).flatten()

alpha = 0.9
beta = 0.9

print("Predictions begin:")

start = time()
pool = Pool(20)
results = pool.map(cf_predict_evaluation_slim, [index for index in test_list_pid])
pool.close()
pool.join()
print('Time taken:', time() - start)

prediction_result = zeros((len(test_list_pid),max_n_predictions+1))
prediction_result[:,0] = test_list_pid

for i in range(len(results)):
    prediction_result[i,range(1,501)] = results[i]

savetxt('../submit/cf2submit_v9.txt', prediction_result, delimiter=',')
