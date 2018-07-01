# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:34:39 2018

@author: bwhe
"""

import math
import scipy.sparse as sp
import numpy
 


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

    
    def Score(self, user_songs,  all_songs, all_songs_uc_spmat):
        # first the intersection
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
        self.tau=1000

    def Add(self,p):
        self.predictors.append(p)

class SReco(Reco):

    '''Implements Aggregated Stochastic Recommender'''

    def __init__(self,_all_songs, _A, threshould, s2uc):
        Reco.__init__(self,_all_songs)
        self.threshould = threshould
        self.Gamma=[]
        self.flag = 0
        self.final_score4calib = {}
        self.all_songs_uc_spmat = []
        self.s2uc = s2uc
        self.A = _A
        


    def RecommendToUser(self, user, u2s_v, test_u2s_v):
        for p in self.predictors:
            ssongs=[]
            # if user in user-song dic, 
            if user in test_u2s_v:                
                res = p.Score(test_u2s_v[user], self.all_songs, self.all_songs_uc_spmat)
                ssongs = numpy.argsort(res)[-(self.tau+100):][::-1]
                self.flag += 1
                if self.flag % 10 == 0:
                    print("get scores for %s playlists"%(self.flag))
            else:
                ssongs=list(self.all_songs)
            
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
        for i,u in enumerate(l_users):
            cleaned_songs, score = self.RecommendToUser(u, u2s_v, test_u2s_v)
            rec4users[i] = cleaned_songs
            scores[i] = score
        return rec4users, scores
