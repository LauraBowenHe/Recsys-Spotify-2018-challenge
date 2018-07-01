# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:06:16 2018

@author: bwhe
"""


import pandas as pd
import numpy as np
import pickle
import gc
from gensim.models import Word2Vec


##############################################################################
### pos_song with song w2v similarity
##############################################################################
def build_track_w2v(w2v_test_pids):
    with open('./train_pid2songs_5.pkl', 'rb') as f:
        pid2songs = pickle.load(f)
    sentences = []    
    for pid, songs in pid2songs.items():
        if pid in w2v_test_pids.tolist():
            continue
        songs = [str(song) for song in songs]
        sentences.append(songs)
    del pid2songs
    gc.collect()
    
    modelname = 'w2v_model1.bin'
    model = Word2Vec(sentences, window=5, negative=10, min_count=1, sample=0.001, workers=4)
    model.save(modelname)
    print("finish train w2v_model1.bin")


##############################################################################
### pos_song album with album similarity
##############################################################################
def build_album_w2v(w2v_test_pids):
    with open('../data/song2album.pkl', 'rb') as f:
        song2album = pickle.load(f)
    
    with open('./train_pid2songs_5.pkl', 'rb') as f:
        pid2songs = pickle.load(f)
    sentences = []
    for pid, songs in pid2songs.items():
        if pid in w2v_test_pids:
            continue
        album = [str(song2album[song]) for song in songs]
        sentences.append(album)
    del pid2songs
    gc.collect()
    
    modelname = 'w2v_model2.bin'
    model = Word2Vec(sentences, window=5, negative=10, min_count=1, sample=0.001, workers=4)
    model.save(modelname)
    print("finish train w2v_model2.bin")


#############################################################################
### pos song artist with artist similarity
#############################################################################
def build_artist_w2v(w2v_test_pids):
    with open('../data/song2artist.pkl', 'rb') as f:
        song2artist = pickle.load(f)
        
    with open('./train_pid2songs_5.pkl', 'rb') as f:
        pid2songs = pickle.load(f)
    sentences = []
    for pid, songs in pid2songs.items():
        if pid in w2v_test_pids:
            continue
        artist = [str(song2artist[song]) for song in songs]
        sentences.append(artist)
    del pid2songs
    gc.collect()
        
    modelname = 'w2v_model3.bin'
    model = Word2Vec(sentences, window=5, negative=10, min_count=1, sample=0.001, workers=4)
    model.save(modelname)
    print("finish train w2v_model3.bin")
