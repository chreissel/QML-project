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

    def __init__(self, layers,error = nn.BCELoss(),lr = 2e-3,lr_decay = 0.04,n_epochs=100):
        super(FFWD, self).__init__()

        # make sure to run on the correct hardware
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device == 'cuda':
            self.batch_size = 10000
        else:
            self.batch_size = 528    
 
        self.error = error
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs


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

    # training procedure
    def train(self, error, optimizer, n_epochs, data):
        self.ffwd.train()
        for epoch in range(1, n_epochs + 1):
            loss_epoch = 0
            batch = 0
            tot_batch = len(data)
            for d in data:
                x,y = d
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = self.ffwd(x)
                loss = error(output, y)
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                batch += 1

                if batch % 10 == 0:
                    print(f'batch {batch}/{tot_batch}')

            loss_epoch = loss_epoch/float(tot_batch)
            if epoch % 1 == 0:
                print(f'############################# epoch {epoch} \t Loss: {loss_epoch:.4g} ##############################')

    def fit(self, X_train, y_train):

        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(y_train)
        dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)

        ffwd = self.ffwd.to(self.device)
        print(ffwd)

        error = nn.BCELoss()
        print("Learning rate: {0}, Learning rate decay: {1}".format(self.lr, self.lr_decay))
        optimizer = optim.Adagrad(ffwd.parameters(),lr=self.lr,lr_decay=self.lr_decay)
        print(ffwd.parameters())

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print("start training now")
        self.train(error=error, optimizer=optimizer, n_epochs=self.n_epochs, data=data_loader)


    def predict(self, X_test):
        X_test_device = torch.from_numpy(X_test).to(self.device)
        with torch.no_grad():
            pred = self.ffwd(X_test_device.float())
        pred = pred.cpu().numpy()
        return pred


