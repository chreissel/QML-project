# load basic packages

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# initilize autoencoder (identical to Vasilis)
class Autoencoder(nn.Module):

    def __init__(self, layers, dropout=False):
        super(Autoencoder, self).__init__()

        # Encoder
        encod_layers = []
        for i in range(len(layers)):
            encod_layers.append(nn.Linear(layers[i], layers[i+1]))

            if i == len(layers) - 2:
                encod_layers.append(nn.Sigmoid())
                break

            encod_layers.append(nn.ELU(True))
            if dropout:
                encod_layers.append(nn.Dropout(0.2))
        self.encode = nn.Sequential(*encod_layers)

        # Decoder
        layers_inv = layers[::-1]
        print(layers_inv)
        decod_layers = []
        for i in range(len(layers)):
            print(i)
            decod_layers.append(nn.Linear(layers_inv[i], layers_inv[i+1]))

            if i == len(layers) - 2:
                decod_layers.append(nn.Sigmoid())
                break

            decod_layers.append(nn.ELU(True))
            if dropout:
                decod_layers.append(nn.Dropout(0.2))
        self.decode = nn.Sequential(*decod_layers)


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

