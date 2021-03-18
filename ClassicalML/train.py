# load basic packages

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from models import FFWD,SVM,BDT
import torch.utils
import torch.utils.data
from sklearn import metrics

# Arguments for functions
import argparse
parser = argparse.ArgumentParser(description="Preselection & Train/Test Split of Samples")
parser.add_argument('--path', help = "path from where to read files")
parser.add_argument('--output', help = "path where to store files")
parser.add_argument('--classifier', help = "classical ML to be used, options: SVM, DNN, BDT")
args = parser.parse_args()
if not args.path.endswith('/'):
    path = args.path + "/"
else:
    path = args.path
if not args.output.endswith('/'):
    output = args.output + "/"
else:
    output = args.output

# Load Dataset
print('Loading Dataset...')
X_train = np.load(path+"X_train.npy")
y_train = np.load(path+"y_train.npy")
X_test = np.load(path+"X_test.npy")
y_test = np.load(path+"y_test.npy")
print('Dataset successfully loaded...')

# Initialization of problem
if args.classifier == "DNN":
    model = FFWD([X_train.shape[1],1024,1024,512,256,128,1])
elif args.classifier == "BDT":
    model = BDT() 
elif args.classifier == "SVM":
    model = SVM(nevents=5000)
print('Model successfully initialized')

# Training 
model.fit(X_train, y_train)
print('Model successfully trained')

# Prediction
pred = model.predict(X_test)
print('Prediction successfully finished')
# check if file exists and delete it
import os
if os.path.exists(output+"pred_"+args.classifier+".npy"):
    os.remove(output+"pred_"+args.classifier+".npy")
np.save(output+"pred_"+args.classifier,pred)

from sklearn import metrics
auc = metrics.roc_auc_score(y_test, pred)
print("AUC store: %f.2" %auc)

# Calculate accuracy
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, pred))

# Calculate AUC with uncertainties
#from sklearn import metrics
#auc = []

#X_valid = np.load(path+"X_valid.npy")
#y_valid = np.load(path+"y_valid.npy")

#for i in range(X_valid.shape[0]):
#    if args.classifier == "BDT":
#        pred = bdt.predict(X_valid[i,:,:])
#    if args.classifier == "DNN":
#        # Prediction
#        X_test_device = torch.from_numpy(X_valid[i,:,:]).to(device)
#        with torch.no_grad():
#            pred = ffwd.ffwd(X_test_device.float())
#        pred = pred.cpu().numpy()

#    auc.append(metrics.roc_auc_score(y_valid, pred))

#import statistics
#mean = statistics.mean(auc)
#stdev = statistics.stdev(auc)

#print("AUC store: {0} +/- {1}".format(mean, stdev))
