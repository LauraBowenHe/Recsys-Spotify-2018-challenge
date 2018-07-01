# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 08:54:07 2018

@author: bwhe
"""

import sys
import os
import ast
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb



def remove_iteral(sentence):
    return ast.literal_eval(sentence) 

###########################################################################
### predict
###########################################################################
version = str(sys.argv[1])
filename = 'cf2xgb_1_pred1_v'+version+'.csv.gz'
model_name = 'model1.txt'
usecols = ['track_uri', 'cf_prob', 'album_uri', 'artist_uri', 'name_sim', 
           'album_sim', 'artist_sim', 'cf_rlt_prob', 'freq', 'w2v_track_sim', 'w2v_album_sim',
           'w2v_artist_sim']
savepath = '../submit/'
nrows = None
result_test = pd.read_csv(filename, nrows=nrows)
result_test['pos_songs'] = result_test['pos_songs'].apply(remove_iteral)
x_test = result_test[usecols].values
#############################################################################
## create dataset for lightgbm
#############################################################################
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 63,
    'max_depth': 5,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': 0
}

print("start predicting...")
bst = lgb.Booster(model_file=model_name)
y_pred = bst.predict(x_test)
result_test['prob'] = y_pred.tolist()
print(result_test['cf_prob'].describe())
print(result_test['prob'].describe())
###########################################################################
##  save prediction
###########################################################################
df = result_test[['pid', 'track_uri', 'prob']]
gc.collect()
df = df.set_index('track_uri').groupby('pid')['prob'].nlargest(550).reset_index()
df['track_uri'] = df['track_uri'].apply(lambda x: str(x))
df = df[['pid', 'track_uri']].groupby('pid')['track_uri'].apply(' '.join).reset_index()

def cov2int(sentence):
    return [int(x) for x in sentence.split(' ')]
df['pred'] = df['track_uri'].apply(cov2int)
pos_name = '../data/test_list_song_v'+version+'.csv.gz'
df_pos = pd.read_csv(pos_name, usecols=['pid', 'songs'])
df_pos['songs'] = df_pos['songs'].apply(remove_iteral)

df = df.merge(df_pos, left_on=['pid'], right_on=['pid'], how='left')
print(df.head())
print(df.shape)
res = np.zeros((df.shape[0], 501))
i = 0
for tup in df.itertuples():
    res[i][0] = tup.pid
    pos_songs = tup.songs
    pred = tup.pred
    cleaned_songs = []
    for song in pred:
        if len(cleaned_songs) >= 500:
            break
        if song not in pos_songs:
            cleaned_songs.append(song)
    res[i][1:] = cleaned_songs
    i += 1

np.savetxt(os.path.join(savepath, 'lgb2submit_v2.txt'), res, delimiter=',')
