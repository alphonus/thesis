#!/usr/bin/python
# -*- coding: utf-8 -*-
#%%
#from importlib import reload 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os, time, random, argparse, timeit
import numpy as np
from itertools import islice
from functools import reduce

import torch

from model import SimpleNet
from instance_loader import GraphDataset
#from util import load_weights, save_weights
#from tabucol import tabucol, test

from torch_geometric.loader import DataLoader

from dataclasses import dataclass
@dataclass
class Config():
    embed_size: int = 64
    timesteps: int = 32
    epochs: int = 10000
    batchsize: int = 8
    path: str = 'adversarial-training'
    seed: int = 42
    lr: float = 0.01



def GNN_GCP(graph, n_colors, config: Config):
    """
    algorithm according to the model
    """
    pass
    # compute binary adjacency matrix from vertex to vertex

    #compute binary adjacency matrix from vertex to colors

    #compute initial vertex embeddings

    #compute initial color embedings
#%%
from torch_geometric.datasets import TUDataset
config = Config
dataset = GraphDataset(root='./adversarial-training/')
#dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=64, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(num_features=dataset.num_node_features,num_classes=dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()


#%%
# Initialize training
train_hist = {}
train_hist['loss'] = []

model.train()
n_epochs = 100
for epoch in range(n_epochs):

    for idx, batch in enumerate(loader):
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y[:,0,None])
        loss.backward()
        optimizer.step()

        train_hist['loss'].append(loss.item())
  
        
    print('[Epoch %4d/%4d] [Batch %4d/%4d] Loss: % 2.2e' % (epoch + 1, n_epochs, 
                                                                idx + 1, len(loader), 
                                                                loss.item()))
        
# %%
