# load basic packages

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from models import Autoencoder
import torch.utils
import torch.utils.data


# training procedure
def train(model, error, optimizer, n_epochs, data):
    model.train()
    for epoch in range(1, n_epochs + 1):
        batch = 0
        tot_batch = len(data)
        for x in data:
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = error(output, x)
            loss.backward()
            optimizer.step()
            batch += 1

            if batch % 20 == 0:
                print(f'batch {batch}/{tot_batch}')

        if epoch % 1 == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')



# make sure to run on the correct hardware
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset and split in train/test sample
X = np.load("/work/creissel/pytorch_tests/inputs.npy")
Y = np.concatenate((np.ones(int(X.shape[0]*0.5)),np.zeros(int(X.shape[0]*0.5))),axis=0) 

X = MinMaxScaler().fit_transform(X)
print("inputs successfully normalized")

# split data
# keep test sample aside
split_inds = np.arange(0, X.shape[0]) #to keep track of which event was splitted to which set
split = train_test_split(split_inds,X,Y,test_size=0.6)
split = [ split[ix] for ix in range(0,len(split),2) ]
inds_train, X_train, y_train = split
print("inputs split in test and train samples")


# Actual training
device = ('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Autoencoder([X.shape[1],64,52,44,32,24,16]).double().to(device)
print(encoder)

error = nn.MSELoss(reduction = 'mean')
optimizer = optim.Adam(encoder.parameters(),lr=2e-3)
print(encoder.parameters())

data_loader = torch.utils.data.DataLoader(X_train, batch_size=128, shuffle=True)
print("start training now")
train(model=encoder, error=error, optimizer=optimizer, n_epochs=1, data=data_loader)


x = torch.from_numpy(X_train).to(device)
with torch.no_grad():
    latent = encoder.encode(x)
    decoded = encoder.decode(latent)
    mse = error(decoded, x).item()
    enc = latent.cpu().detach().numpy()
    dec = decoded.cpu().detach().numpy()

print(f'Root mean squared error: {np.sqrt(mse):.4g}')

fig = plt.figure()
plt.hist(decoded[:,0],range = (0,1), bins = 100,histtype='step',color='red', density=True)
plt.hist(X[:,0],range = (0,1), bins = 100,histtype='step',color='blue', density=True)
plt.show()
