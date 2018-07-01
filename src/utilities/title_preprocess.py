# -*- coding: utf-8 -*-
"""
reference:
    https://www.kaggle.com/fizzbuzz/toxic-data-preprocessing/code

Created on Fri Jun  1 18:00:22 2018

@author: bwhe
"""

import pandas as pd
import copy
import re


class BaseTokenizer(object):
    def process_text(self, text):
        raise NotImplemented

    def process(self, texts):
        for text in texts:
            yield self.process_text(text)

RE_PATTERNS = {'party':['party']}

class PatternTokenizer(BaseTokenizer):
    def __init__(self, lower=True, initial_filters="[^a-z!@#\$%\^\&\*_\-,\.']", patterns=RE_PATTERNS,
                 remove_repetitions=True):
        self.lower = lower
        self.patterns = patterns
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions

    def process_text(self, text):
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x.split()

    def process_ds(self, ds):
        ### ds = Data series

        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()

        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
            ds = ds.str.replace(pattern, r"\1")
    
        ds = ds.str.replace(r"[^a-z0-9' ]", '')
        
        
        return ds.str.split()

def build_clean_title(readfile, savefile):
    print("raed file ... %s"%(readfile))
    df = pd.read_csv(readfile)  
    tokenizer = PatternTokenizer()
    df["more_clean_name"] = tokenizer.process_ds(df["name"]).str.join(sep=" ")
    df.to_csv(readfile, index=False, compression='gzip')
    
    import pickle
    print("save pid clean name ...")
    test_pid2cleanname = {}
    for i in range(df.shape[0]):
        test_pid2cleanname[df['pid'][i]] = df['more_clean_name'][i] 
    
    with open(savefile, 'wb') as f:
        pickle.dump(test_pid2cleanname, f)
        
        
####################### train title ##########################
readfile = '../data/train_list_info.csv.gz'
savefile = '../data/pid2more_clean_name.pkl'
build_clean_title(readfile, savefile)

####################### test title ###########################
readfile = '../data/test_list_info.csv.gz'
savefile = '../data/test_pid2more_clean_name.pkl'
build_clean_title(readfile, savefile)