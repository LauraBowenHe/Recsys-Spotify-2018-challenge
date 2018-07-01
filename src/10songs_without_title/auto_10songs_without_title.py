import os

os.system("python lgb_train.py")
os.system("python prediction.py 6 10")
os.system("python lgb_test_features.py")
os.system("python lgb_predict.py 6")
