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
model_name = 'model11_0630.txt'



result_train = pd.read_csv(savefile_train)
result_test = pd.read_csv(savefile_test)
result_train['pos_songs'] = result_train['pos_songs'].apply(remove_iteral)
result_test['pos_songs'] = result_test['pos_songs'].apply(remove_iteral)

print("get result train and test...")
pids = result_test.pid
df = pd.read_csv(readfile, usecols=['pid','real'], nrows=nrows)
final_df_test = df[df['pid'].isin(pids)]
final_df_test['real'] = final_df_test['real'].apply(remove_iteral)
del df
gc.collect()
print("get pid...")

''' remove pid '''
with open('result.txt', 'a') as f:
    f.write("recover pid -> %s\n"%(model_name))


usecols = ['track_uri', 'cf_prob', 'album_uri', 'artist_uri', 'name_sim', 'album_sim', 'artist_sim',
           'cf_rlt_prob', 'w2v_track_sim', 'w2v_album_sim','w2v_artist_sim']
y_train = result_train['label'].values
x_train = result_train[usecols].values
y_test = result_test['label'].values
x_test = result_test[usecols].values
##############################################################################
### create dataset for lightgbm
##############################################################################
lgb_train = lgb.Dataset(x_train, y_train)

lgb_test = lgb.Dataset(x_test, y_test) 

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

print("start training...")
start_time = time.time()
gbm = lgb.train(params, 
                lgb_train,
                num_boost_round=1700,
                valid_sets=[lgb_train, lgb_test],
                early_stopping_rounds=5)

print("lgb train --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
gbm.save_model(model_name)
print("model save --- %s seconds ---" % (time.time() - start_time))

print("start predicting...")
y_train_pred = gbm.predict(x_train)
y_pred = gbm.predict(x_test)

result_test['prob'] = y_pred.tolist()
result_train['prob'] = y_train_pred.tolist()


del lgb_train, lgb_test, gbm
gc.collect()
############################################################################
###  evaluation metric
############################################################################

df = result_test[['pid', 'track_uri', 'prob']]
gc.collect()
df = df.set_index('track_uri').groupby('pid')['prob'].nlargest(500).reset_index()
df['track_uri'] = df['track_uri'].apply(lambda x: str(x))
df = df[['pid', 'track_uri']].groupby('pid')['track_uri'].apply(' '.join).reset_index()
print(df.shape)

print("load test...")  
final_df_test = final_df_test.merge(df, left_on=['pid'], right_on=['pid'], how='left')
print(final_df_test.shape)
def cov2int(sentence):
    return [int(x) for x in sentence.split(' ')]
final_df_test['pred'] = final_df_test['track_uri'].apply(cov2int)

print("the test shape is %s"%(final_df_test.shape[0]))


print("evaluation metric...")
with open('result.txt', 'a') as f:
    f.write("lgb metric values...\n")
    final_df_test['r_precision'] = final_df_test.apply(metric.r_precision, axis=1).tolist()
    r_precision_res = np.array(final_df_test.apply(metric.r_precision, axis=1).tolist())
    print("r_precision is %s"%(np.mean(r_precision_res)))
    f.write("r_precision is %s"%(np.mean(r_precision_res)))
    
    final_df_test['ndcg'] = final_df_test.apply(metric.ndcg, axis=1).tolist()
    ndcg_res = np.array(final_df_test.apply(metric.ndcg, axis=1).tolist())
    print("ndcg is %s"%(np.mean(ndcg_res)))
    f.write("ndcg is %s \n"%(np.mean(ndcg_res)))
    
    final_df_test['pclc'] = final_df_test.apply(metric.playlist_extender_clicks, axis=1).tolist()
    playclick_res = np.array(final_df_test.apply(metric.playlist_extender_clicks, axis=1).tolist())
    print("play click is %s"%(np.mean(playclick_res)))
    
    f.write("play click is %s"%(np.mean(playclick_res)))

