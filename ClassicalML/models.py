# load basic packages

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# Standard SVM
class SVM():

    def __init__(self, nevents):
        #self.clf = svm.SVC(gamma=2, C=1, probability = True)
        #self.nevents = nevents
        self.clf = SGDClassifier(loss="log", alpha=10.0,verbose=1, warm_start=True)
        
    def fit(self, X, y):

        batchsize = 2000
        batches = np.arange(0,X.shape[0],batchsize)
        
        for i in range(len(batches)):
            if i == (len(batches) - 1):
                self.clf.partial_fit(X[batches[i]:],y[batches[i]:], classes=np.unique(y))
            else:
                self.clf.partial_fit(X[batches[i]:batches[i+1]],y[batches[i]:batches[i+1]], classes=np.unique(y))
            if i % 100 == 0:    
                print("Batch %i processed" %i)

        """
        X_selc = X[:self.nevents, :]
        y_selc = y[:self.nevents]

        import pdb
        pdb.set_trace()

        self.clf.fit(X_selc, y_selc)
        """

    def predict(self,X):
        y_pred = self.clf.predict_proba(X)
        return y_pred[:,1]
        
# Standard Boosted Decision Tree
class BDT():

    def __init__(self):
        #self.clf = RandomForestClassifier(n_estimators=1, verbose=1, max_depth=2, warm_start=True)
        self.clf = GradientBoostingClassifier(n_estimators=1, verbose=1, max_depth=2, warm_start=True)

    def fit(self, X, y):

        batchsize = 2000
        batches = np.arange(0,X.shape[0],batchsize)

        for i in range(len(batches)):
            if i == (len(batches) - 1):
                self.clf.fit(X[batches[i]:],y[batches[i]:])
            else:
                self.clf.fit(X[batches[i]:batches[i+1]],y[batches[i]:batches[i+1]])
            self.clf.n_estimators += 1
            if i % 100 == 0:
                print("Batch %i processed" %i)

    def predict(self,X):
        y_pred = self.clf.predict_proba(X)
        return y_pred[:,1]


# FFWD DNN (identical settings as before, but now in pytorch)
class FFWD(nn.Module):

    def __init__(self, layers):
        super(FFWD, self).__init__()

        torch_layers = []

        torch_layers.append(nn.BatchNorm1d(layers[0]))

        for i in range(len(layers)):
            torch_layers.append(nn.Linear(layers[i], layers[i+1]))

            if not i == len(layers)-2:
                torch_layers.append(nn.BatchNorm1d(layers[i+1]))
                torch_layers.append(nn.Dropout(0.5))

            if i == len(layers) - 2:
                torch_layers.append(nn.Sigmoid())
                break

            torch_layers.append(nn.LeakyReLU(0.2))
            #torch_layers.append(nn.PReLU())
        self.ffwd = nn.Sequential(*torch_layers)

    def forward(self, x):
        x = self.ffwd(x)
        return x

