import sys
import os, ast
import gc
import pandas as pd
import numpy as np

from gensim import models


####################################################################################################
# evaluation metric
####################################################################################################
def r_precision(df):
    pred = df['pred']
    real = df['real']
    intersct_len = 1. * len(list(set(pred)&set(real)))
    return intersct_len / len(real) 

def idcg(pred, real):
    intersct_len = len(list(set(pred) & set(real)))
    if intersct_len == 0:
        return 1
    for i in range(intersct_len):
        if i == 0:
            idcg = 1.
        else:
            idcg += 1. / (np.log2(i+1))
    return idcg

def dcg(pred, real):
    if len(list(set(pred)&set(real))) == 0:
        return 0.
    
    for pos, item in enumerate(pred):
        if pos+1 == 1:
            if item in real:
                dcg = 1.
            else:
                dcg = 0.
        else:
            if item in real:
                dcg += 1. / (np.log2(pos+1))
    return dcg
                
def ndcg(df):
    pred = df['pred']
    real = df['real']
    return dcg(pred, real) / idcg(pred, real)


####################################################################################################
# load model
####################################################################################################
version = sys.argv[1]
modelname = 'song_v'+str(version)+'.bin'
model = models.Word2Vec.load(modelname)


####################################################################################################
# load test data
####################################################################################################
__path__ = '../data/'
testdata = 'train_list_song_info.csv.gz'
skiprows = 59711785
dtypes = {'track_uri': str}
test = pd.read_csv(os.path.join(__path__, testdata), skiprows=skiprows+1, names=['pos', 'track_uri', 'pid'], dtype=dtypes, nrows=10000)
print("read test...")

all_pid = test.pid.unique()
list_song = test.groupby('pid').apply(lambda x: list(x.track_uri))
pids = []
real = []

pid_len = len(all_pid)
songs = np.zeros([pid_len, 500])
pos = 0
for pid in all_pid:
    flag = 0
    res = list_song[pid]
    for i in range(len(res)):
        try:
            songs[pos] = [int(x[0]) for x in model.most_similar(positive=res[i], topn=500)]
            pos += 1
            flag = 1
            break
        except:
            continue
    # there is not match vector in this list, then skip it
    if flag == 0:
        continue
    del res[i]    
    pids.append(pid)    
    real.append([int(x) for x in res])
del test, all_pid, list_song
gc.collect()

songs = list(songs[0:pos])
df = pd.DataFrame({'pid':pids, 
                   'pred': songs,
                   'real': real})
df.to_csv('../data/songs_seed'+str(version)+'.csv.gz', compression='gzip')
print("predict song...")

# df['real'] = df['real'].map(ast.literal_eval)
# df['pred'] = df['pred'].map(ast.literal_eval)
r_precision_res = np.array(df.apply(r_precision, axis=1).tolist())
print("r_precision is %s"%(np.mean(r_precision_res)))
ndcg_res = np.array(df.apply(ndcg, axis=1).tolist())
print("ndcg is %s"%(np.mean(ndcg_res)))
res_df = pd.DataFrame({'r_precision': r_precision_res,
                       'ndcg': ndcg_res})
res_df.to_csv('../data/result_songs_v'+str(version)+'.csv.gz', compression='gzip')
#print(r_precision(real, pred))
#print(ndcg(pred, real))
if not os.path.exists('../res/'):
    os.mkdir('../res/')

with open('../res/result_song_v'+str(version)+'.txt', 'w') as fp:
    fp.write("r_precision is %s\n"%(np.mean(r_precision_res)))
    fp.write("ndcg is %s\n"%(np.mean(ndcg_res)))
