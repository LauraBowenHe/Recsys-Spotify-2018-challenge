# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:52:43 2018

@author: bwhe
"""

import os

os.system("python pre_process_resys.py")
os.system("python title_preprocess.py")
os.system("python gen_sub_task_pids.py")
os.system("python song2freq.py")
os.system("python song2attributes.py")
