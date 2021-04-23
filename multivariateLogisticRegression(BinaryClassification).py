# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 21:14:15 2021

@author: jws
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# dataset 정의
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
        self.y_data = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x, y
# dataset 선언
dataset = CustomDataset()
# dataloader 선언
dataloader = DataLoader(
    dataset, 
    batch_size = 2,
    shuffle=True
)

# 모델 정의. LogisticRegression, BinaryClassification
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        nn.init.constant_(self.linear.weight, 0)
        self.sigmoid = nn.Sigmoid()  # 출력을 좀 더 0과 1에 가깝게 만들어준다
        
    def forward(self, x):
        y = self.linear(x)
        return self.sigmoid(y)
# 모델 선언
model = BinaryClassifier()
# optimizer 선언
optimizer = optim.SGD(model.parameters(), lr = 0.1)


# train
epochs = 100
for epoch in range(epochs+1):
    for i, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        hypothesis = model(x_train)
        
        cost = F.binary_cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])   # 0.5를 기준으로 작으면 0, 크면 1로 한다.
        correct_prediction = prediction.float() == y_train    # y_train과 비교하여 같게 예측한것만 1, 틀린것은 0으로 한다.
        accuracy = correct_prediction.sum().item() / len(correct_prediction)   # 맞게 예측한 것 / 전체 하여 정답률을 구한다.
        print("epoch:{}/{} =========, \ncost: {}, \naccuracy:{}%, \ny_train:{}, \nprediction:{} \n".format(
                epoch, epochs, cost.item(), accuracy*100, y_train.detach().squeeze(), prediction.squeeze()))
       
        
# test
x_test = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y_test = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])

hypothesis = model(x_test)

prediction = hypothesis >= 0.5
correct_prediction = prediction == y_test
accuracy = correct_prediction.sum().item() / len(correct_prediction)

print("y_test:", y_test.squeeze())
print("prediction:", correct_prediction.squeeze())
print("accuracy:", accuracy)   