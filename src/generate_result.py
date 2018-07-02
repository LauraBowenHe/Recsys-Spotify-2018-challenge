# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:39:45 2018

@author: bwhe
"""


import os

if not os.path.exists('./data/'):
    os.mkdir('./data/')

os.chdir('./utilities')
os.system("python auto_generate.py")

os.chdir('./title_only')
os.system("python prediction.py")

os.chdir('./1song')
os.system("python auto_1songs.py")

os.chdir('./5songs_with_title')
os.system("python auto_5songs_title.py")

os.chdir('./5songs_without_title')
os.system("python auto_5songs_without_title.py")

os.chdir('./10songs_with_title')
os.system("python auto_10songs_title.py")

os.chdir('./10songs_without_title')
os.system("python auto_10songs_without_title.py")

os.chdir('./25songs_order')
os.system("python cf_test_submission_25_order.py")

os.chdir('/25songs_shuffle')
os.system("python cf_test_submission_25_shuffle.py")

os.chdir('./100songs_order')
os.system("python cf_test_submission_100_order.py")

os.chdir('./100songs_order')
os.system("python cf_test_submission_100_shuffle.py")

os.chdir('./submit')
os.system("python submit.py")
