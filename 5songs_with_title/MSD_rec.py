# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:34:39 2018

@author: bwhe
"""

import pickle
import os,sys,random,math,time
import gc
import functools
import MSD_util
import scipy.sparse as sp
import numpy
from multiprocessing import Pool
 

def fl():
    sys.stdout.flush()

#l_rec: list of recommended songs
#u2s: mapping users to songs
#tau: 500
def AP(l_rec, sMu, tau):

    np=len(sMu)
    #print "np:", np
    nc=0.0
    mapr_user=0.0
    for j,s in enumerate(l_rec):
        if j>=tau:
            break
        if s in sMu:
        #print "s in sMu"
            nc+=1.0
            mapr_user+=nc/(j+1)
    mapr_user/=min(np,tau)
    return mapr_user

#l_users: list of users
#l_rec_songs: list of lists, recommended songs for users
#u2s: mapping users to songs
#tau: 500
def mAP(l_users, l_rec_songs, u2s, tau):
    mapr=0
    n_users=len(l_users)
    for i,l_rec in enumerate(l_rec_songs):
        if not l_users[i] in u2s:
            continue
        mapr+=AP(l_rec,u2s[l_users[i]], tau)
    return mapr/n_users

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
        
    def printati(self):
        print("PredSI(A=%f,Q=%f)"%(self.A,self.Q))
        
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


class PredSIc(PredSI):
    '''Implement calibrated song-similarity based predictor''' 

    def __init__(self, _s2u_tr, _A=0, _Q=1, f_hsongs=""):
        PredSI.__init__(self, _s2u_tr, _A, _Q)
        self.hsongs={}
        with open('songs_calibrated_score_10.pkl',"rb") as f:
            self.hsongs = pickle.load(f)
        self.THETA = 0.5

    def select_theta(self,h):
        return self.THETA
        
    def calibrate(self, sco, song):
        h = self.hsongs[song]
        theta = self.select_theta(h)
        prob=sco
        if sco<h:
            prob = theta*sco/h
        elif sco>h:
            prob = theta+(1.0-theta)*(sco-h)/(1.0-h)
        return prob

    def Score(self, user_songs,  all_songs):
        np = len(user_songs)
        s_scores={}
        for s in all_songs:
            s_scores[s]=0.0
            if not (s in self.s2u_tr):
                continue
            for u_song in user_songs:
                if not (u_song in self.s2u_tr):
                    continue
                s_match=self.Match(s,u_song)
                s_scores[s]+=math.pow(s_match,self.Q)/np
        for s in all_songs:
            if s in self.hsongs:
                s_scores[s]=self.calibrate(s_scores[s],s)
            else:
                s_scores[s]=0.0
        return s_scores        
    
class PredSU(Pred):

    '''Implement user-similarity based predictor'''
    
    def __init__(self, _u2s_tr, _A=0, _Q=1):
        Pred.__init__(self)
        self.u2s_tr = _u2s_tr
        self.Q = _Q
        self.A = _A


    def Score(self,user_songs,  all_songs):
        s_scores={}
        for u_tr in self.u2s_tr:
            if not u_tr in self.u2s_tr:
                continue
            w=float(len(self.u2s_tr[u_tr] & set(user_songs)))
            if w > 0:
                l1=len(user_songs)
                l2=len(self.u2s_tr[u_tr])
                w/=(math.pow(l1,self.A)*(math.pow(l2,(1.0-self.A))))
                w=math.pow(w,self.Q)
            for s in self.u2s_tr[u_tr]:
                if s in s_scores:
                    s_scores[s]+=w
                else:
                    s_scores[s]=w
        return s_scores

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
        

    def GetStocIndex(self,n,distr):
        r=random.random()
        for i in range(n):
            if r<distr[i]:
                return i
            r-=distr[i]
        return 0
        
    def GetStochasticRec(self,songs_sorted, distr):
        nPreds=len(self.predictors)
        r=[]
        ii = [0]*nPreds
        while len(r)<self.tau:
            pi = self.GetStocIndex(nPreds,distr)
            s = songs_sorted[pi][ii[pi]]
            if not s in r:
                r.append(s)
            ii[pi]+=1
        return r

    def Valid(self, T, users_te, u2s_v, u2s_h, n_batch=10):
        ave_AP=0.0
        for t in range(T):
            rusers = users_te[t*n_batch:(t+1)*n_batch]
            rec=[]
            start=time.clock()
            for i,ru in enumerate(rusers):
                if ru in u2s_v:
                    print("%d] scoring user %s with %d songs"%(i,ru,len(u2s_v[ru])))
                else:
                    print("%d] scoring user %s with 0 songs"%(i,ru))
                fl()
                songs_sorted=[]
                for p in self.predictors:
                    ssongs=[]
                    if ru in u2s_v:
                        ssongs=MSD_util.sort_dict_dec(p.Score(u2s_v[ru],self.all_songs))
                    else:
                        ssongs=list(self.all_songs)
                   
                    cleaned_songs = []
                    for x in ssongs:
                        if len(cleaned_songs)>=self.tau: 
                            break
                        if ru not in u2s_v or x not in u2s_v[ru]:
                             cleaned_songs.append(x)
                                            
                    songs_sorted+= [cleaned_songs]
                    
                rec += [self.GetStochasticRec(songs_sorted, self.Gamma)]

            cti=time.clock()-start
            print("Processed in %f secs"%cti)
            fl()
            # valuta la rec cn la map
            map_cur = mAP(rusers,rec,u2s_h,self.tau)
            ave_AP+=map_cur
            print("MAP(%d): %f (%f)"%(t,map_cur,ave_AP/(t+1)))
            print
            fl()
    
        print("Done!")

    def RecommendToUser(self, user, u2s_v, test_u2s_v):
        for p in self.predictors:
            ssongs=[]
            # if user in user-song dic, 
            if user in test_u2s_v:                
                # p.score will return a dictionary, p[song] = [a similarity score for all songs]
                # get the songs by similarity in descending way
#                score4calib = p.Score(u2s_v[user], self.all_songs)  
#                if self.flag == 0:
#                    self.final_score4calib = score4calib
#                else:
#                    self.final_score4calib = {k:score4calib[k]+self.final_score4calib[k] for k in self.final_score4calib}                
                res = p.Score(test_u2s_v[user], self.all_songs, self.all_songs_uc_spmat)
                ssongs = numpy.argsort(res)[-(self.tau+100):][::-1]
                self.flag += 1
                if self.flag % 10 == 0:
                    print("----------------debug--------------")
                    print(self.flag)
            else:
                ssongs=list(self.all_songs)
            
            cleaned_songs = []
            
            for x in ssongs:
                if len(cleaned_songs)>=self.tau:
                    break
#                print("----------------------debug helper-------------------")
#                print(type(u2s_v[user]))
#                print(u2s_v[user])
#                print("----------------------debug helper-------------------")
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
#            print("----------------------debug helper-------------------")
#            print(u)
#            print("----------------------debug helper-------------------")
            cleaned_songs, score = self.RecommendToUser(u, u2s_v, test_u2s_v)
            rec4users[i] = cleaned_songs
            scores[i] = score
#        self.final_score4calib = {k:self.final_score4calib[k]*1./self.flag for k in self.final_score4calib}
#        song_cali_score_fname = 'songs_calibrated_score_'+str(self.threshould)+'.pkl'
#        with open(song_cali_score_fname, 'wb') as f:
#            pickle.dump(self.final_score4calib, f)
        return rec4users, scores
