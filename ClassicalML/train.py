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
parser.add_argument('--output', help = "path from where to store files")
parser.add_argument('--classifier', help = "classical ML to be used, options: SVM, DNN")
args = parser.parse_args()
if not args.path.endswith('/'):
    path = args.path + "/"
else:
    path = args.path
if not args.output.endswith('/'):
    output = args.output + "/"
else:
    output = args.output


# training procedure
def train(model, error, optimizer, n_epochs, data):
    model.train()
    for epoch in range(1, n_epochs + 1):
        loss_epoch = 0
        batch = 0
        tot_batch = len(data)
        for d in data:
            x,y = d
            x = x.to(device)
            y = y.to(device)
 
            optimizer.zero_grad()
            output = model(x)
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


# Load Dataset
print('Loading Dataset...')
X_train = np.load(path+"X_train.npy")
y_train = np.load(path+"y_train.npy")
X_test = np.load(path+"X_test.npy")
y_test = np.load(path+"y_test.npy")
print('Dataset successfully loaded...')

if args.classifier == "DNN":

    # make sure to run on the correct hardware
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.Tensor(y_train)
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)

    # Actual training
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    ffwd = FFWD([X_train.shape[1],1024,1024,512,256,128,1]).to(device)
    #ffwd = FFWD([X_train.shape[1],256,256,128,64,1]).to(device)
    print(ffwd)

    error = nn.BCELoss()
    #optimizer = optim.Adam(ffwd.parameters(),lr=1e-5)
    lr = 2e-3
    lr_decay = 0.04
    print("Learning rate: {0}, Learning rate decay: {1}".format(lr, lr_decay))
    optimizer = optim.Adagrad(ffwd.parameters(),lr=lr,lr_decay=lr_decay)
    print(ffwd.parameters())

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)
    print("start training now")
    train(model=ffwd, error=error, optimizer=optimizer, n_epochs=100, data=data_loader)


    # Prediction
    X_test_device = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        pred = ffwd.ffwd(X_test_device.float())
    pred = pred.cpu().numpy()

if args.classifier == "SVM":

    svm = SVM(nevents=5000)
    print("SVM successfully initialized")
    svm.fit(X_train, y_train)
    print("SVM successfully trained")
    pred = svm.predict(X_test) 

if args.classifier == "BDT":

    bdt = BDT()
    print("BDT successfully initialized")
    bdt.fit(X_train, y_train)
    print("BDT successfully trained")
    pred = bdt.predict(X_test)

# check if file exists and delete it
import os
if os.path.exists(output+"pred_"+args.classifier+".npy"):
    os.remove(output+"pred_"+args.classifier+".npy")
np.save(output+"pred_"+args.classifier,pred)
#pred = np.where(pred > 0.5, 1,0)

# Calculate accuracy
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, pred))

# Calculate AUC with uncertainties
from sklearn import metrics
auc = []

#pred_kfold = np.array_split(pred,5,axis=0)
#true_kfold = np.array_split(y_test,5,axis=0)

X_valid = np.load(path+"X_valid.npy")
y_valid = np.load(path+"y_valid.npy")

#for i in range(5):
#    auc.append(metrics.roc_auc_score(true_kfold[i], pred_kfold[i]))

for i in range(X_valid.shape[0]):
    if args.classifier == "BDT":
        pred = bdt.predict(X_valid[i,:,:])
    if args.classifier == "DNN":
        # Prediction
        X_test_device = torch.from_numpy(X_valid[i,:,:]).to(device)
        with torch.no_grad():
            pred = ffwd.ffwd(X_test_device.float())
        pred = pred.cpu().numpy()

    auc.append(metrics.roc_auc_score(y_valid, pred))

import statistics
mean = statistics.mean(auc)
stdev = statistics.stdev(auc)

print("AUC store: {0} +/- {1}".format(mean, stdev))
