# -*- coding: utf-8 -*-
"""
This script manage all slice files
for final submit

Created on Tue May 15 10:39:10 2018

@author: bwhe
"""

import os
import gc
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np




__path__ = '.'


# convert song label to song uri
with open('../data/songlabel2uri.pkl', 'rb') as f:
    dic = pickle.load(f)
    
def label2uri(label_list):
    res = [dic[int(x)][0] for x in label_list]
    return res
    

df1 = np.genfromtxt(os.path.join(__path__, 'tosubmit_v1.csv'), delimiter=',').astype(np.int64)
df2 = np.genfromtxt(os.path.join(__path__, 'lgb2submit_v2.txt'), delimiter=',').astype(np.int64)
df1 = np.concatenate((df1, df2), axis=0)
df2 = np.genfromtxt(os.path.join(__path__, 'lgb2submit_v3.txt'), delimiter=',').astype(np.int64)
df1 = np.concatenate((df1, df2), axis=0)
SLICE_NUM = 1
for i in range(SLICE_NUM):
    i = i + 4
    filename = 'cf2submit_v'+str(i)+'.txt'
    print(filename)
    df2 = np.genfromtxt(os.path.join(__path__, filename), delimiter=',').astype(np.int64)
    df1 = np.concatenate((df1, df2), axis=0)
for i in range(2):
    i = i + 5
    filename = 'lgb2submit_v'+str(i)+'.txt'
    print(filename)
    df2 = np.genfromtxt(os.path.join(__path__, filename), delimiter=',').astype(np.int64)
    df1 = np.concatenate((df1, df2), axis=0)
  
for i in range(4):
    i = i + 7
    filename = 'cf2submit_v'+str(i)+'.txt'
    print(filename)
    df2 = np.genfromtxt(os.path.join(__path__, filename), delimiter=',').astype(np.int64)
    df1 = np.concatenate((df1, df2), axis=0)
    

res = np.zeros((df1.shape[0], 501), dtype=object)    
for i in range(df1.shape[0]):
    res[i][0] = df1[i][0]
    res[i][1:] = label2uri(df1[i][1:])

print("finish saving...")


df = pd.DataFrame(res)
df.to_csv(os.path.join(__path__, 'submit.csv'), sep=',', index=False)
