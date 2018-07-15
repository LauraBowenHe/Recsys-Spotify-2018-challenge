import sys
import gc
import pandas as pd
import os
from gensim.models import Word2Vec


window = int(sys.argv[1])
negative = int(sys.argv[2])
version = sys.argv[3]


""" import training data """

sentences = []
dtypes = {'track_uri': str}
nrows = 59711785
song = pd.read_csv('../data/train_list_song_info.csv.gz', dtype=dtypes, nrows=nrows)
all_pid = song.pid.unique()
list_song = song.groupby('pid').apply(lambda x: list(x.track_uri))
for pid in all_pid:
    sentences.append(list_song[pid])
#print(sentences)
#print(type(sentences))
del all_pid, list_song
gc.collect()
print("finish reading...")

""" train data """
model = Word2Vec(sentences, window=window, negative=negative, min_count = 1, sample=0.001)
# print(list(model.wv.vocab))
print("finish train...")

""" save model """
modelname = 'song_v'+str(version)+'.bin'
model.save(modelname)
