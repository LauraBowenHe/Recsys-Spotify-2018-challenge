# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:04:14 2018

@author: bwhe
"""


import pickle
import os,sys,random,math,time
import gc
import functools
import scipy.sparse as sp
from scipy.sparse import coo_matrix 
import numpy  
from multiprocessing import Pool
from sklearn.svm import LinearSVC


###
### PREDICTORS
###

class Pred:
    '''Implement generic predictor'''        
    
    def __init__(self):
        pass

    def Score(self,user_songs,  all_songs):
        return {}

class PredSI(Pred):
    '''Implement song-similarity based predictor'''
    def __init__(self, _s2u_tr, _A, _Q, mat, coomat, s2uc):
        Pred.__init__(self)
        self.s2u_tr = _s2u_tr
        self.Q = _Q
        self.A = _A
        self.mat = mat  
        self.coomat = coomat
        self.s2uc = s2uc
        self.u2smat = coomat.transpose()
        
        
    def Match(self, u_song):
        s = self.the_song
        # number of users has listened to song s
        l1=len(self.s2u_tr[s])
        # number of users has listened to song u
        l2=len(self.s2u_tr[u_song])
        # the intersections of users listened both songs
        up = float(len(self.s2u_tr[s]&self.s2u_tr[u_song]))
        if up>0:
            dn = math.pow(l1,self.A)*math.pow(l2,(1.0-self.A))
            return math.pow(up/dn, self.Q)
        return 0.0
##############################################################################
### cacluate similarity of each song in the all song dataset  between the user songs
### return all songs' score as an array
#############################################################################
    def Score(self, user_songs,  all_songs, all_songs_uc_spmat):
        # first the intersection
#        print(user_songs)
#        print(self.s2uc[user_songs[0]])
        usong_mat = sp.vstack([self.mat.getrow(x) for x in user_songs]).tocoo()
        inter_mat = (self.coomat).dot(usong_mat.transpose()).todense()
        
        # second step the song pid counts
        song_pidc = all_songs_uc_spmat
        # thrid step the usong pid count
        usong_pidc = numpy.array([(math.pow(self.s2uc[usong], (1.0-self.A))) for usong in user_songs])
        dn = numpy.outer(song_pidc, usong_pidc)
        res = numpy.divide(inter_mat, dn)
        res[numpy.isnan(res)] = 0
        res = numpy.asarray(res)
        res = numpy.power(res, self.Q)
        res = numpy.squeeze(numpy.asarray(numpy.mean(res, axis=1)))
        return res
    
###
### RECOMMENDERS
###

class Reco:

    '''Implements Recommender'''

    def __init__(self, _all_songs):
        self.predictors=[]
        self.all_songs=_all_songs
        self.tau = 1000

    def Add(self,p):
        self.predictors.append(p)

class SReco(Reco):

    '''Implements Aggregated Stochastic Recommender'''

    def __init__(self,_all_songs, _A, threshould, s2uc, u2s_v):
        Reco.__init__(self,_all_songs)
        self.threshould = threshould
        self.Gamma=[]
        self.flag = 0
        self.final_score4calib = {}
        self.all_songs_uc_spmat = []
        self.s2uc = s2uc
        self.A = _A
        self.u2s_v = u2s_v
        self.all_user_song_count = []
        self.all_songs_uc_spmat = numpy.zeros(2262292)
        for song in (self.all_songs):
            self.all_songs_uc_spmat[song] = math.pow(self.s2uc[song], self.A)
        
  
    
    def RecommendToUser(self, user, test_u2s_v):
        for p in self.predictors:
            ssongs = []
            if user in test_u2s_v:                
                res = p.Score(test_u2s_v[user], self.all_songs, self.all_songs_uc_spmat)
                ssongs = numpy.argsort(res)[-(self.tau+100):][::-1]  
                self.flag += 1
                print("------------%s---------"%(self.flag))
            else:
                print("==========================should never come here==========================")
                ssongs = list(self.all_songs)
            
            cleaned_songs = []
            
            for x in ssongs:
                if len(cleaned_songs)>=self.tau:
                    break
                if x not in test_u2s_v[user]:
                    cleaned_songs.append(x)        
            
            scores = [res[song] for song in cleaned_songs]
                
        return cleaned_songs, scores
    
    def RecommendToUsers(self, l_users, u2s_v, test_u2s_v):
        rec4users = numpy.zeros((len(l_users), self.tau))
        scores = numpy.zeros((len(l_users), self.tau))
        self.all_songs_uc_spmat = numpy.zeros(2262292)
        for song in (self.all_songs):
            self.all_songs_uc_spmat[song] = math.pow(self.s2uc[song], self.A)
        self.all_user_song_count = numpy.zeros(1000000)
        for user in u2s_v:
            try:
                self.all_user_song_count = math.pow(len(u2s_v[user]), self.U_A)
            except:
                self.all_user_song_count = 0
        for i,u in enumerate(l_users):
            cleaned_songs, score = self.RecommendToUser(u, test_u2s_v)
            rec4users[i] = cleaned_songs
            scores[i] = score
        return rec4users, scores