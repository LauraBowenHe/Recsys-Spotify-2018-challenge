import os

os.system("python main.py")
os.system("python lgb_train_features.py")
os.system("python lgb_train.py")

os.system("python prediction.py 3 5")
os.system("python lgb_test_features.py")
os.system("python lgb_predict.py 3")
