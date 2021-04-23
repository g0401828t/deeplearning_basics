# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 21:38:39 2021

@author: jws
"""

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor([[1, 2, 1, 1],
                                         [2, 1, 3, 2],
                                         [3, 1, 3, 4],
                                         [4, 1, 5, 5],
                                         [1, 7, 5, 5],
                                         [1, 2, 5, 6],
                                         [1, 6, 6, 6],
                                         [1, 7, 7, 7]])
        self.y_data = torch.LongTensor([[2], [2], [2], [1], [1], [1], [0], [0]])
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])
        
        return x, y
dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
    )


class SoftmaxClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassificationModel()

optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 1000
for epoch in range(epochs+1):
    for i, samples in enumerate(dataloader):
        x_train, y_train = samples
        y_train = y_train.squeeze()   # cross_entropy에 넣으려면 펴야한다.
        
        hypothesis = model(x_train)
        
        cost = F.cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        correct_prediction = hypothesis.max(1)[1] == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("epochs: {}/{}, cost: {}, accuracy: {}%".format(
            epoch, epochs, cost.item(), accuracy*100))