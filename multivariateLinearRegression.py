# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:11:10 2021

@author: jws
"""

import torch
from torch.utils.data import Dataset # dataset 형성을 위한
from torch.utils.data import DataLoader # 미니배치를 위한

import torch.nn as nn
import torch.nn.functional as F

# dataset 형성
class CustomDataset(Dataset):
	def __init__(self):
		self.x_data = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70],
							[89, 96, 90]])
		self.y_data = torch.FloatTensor([[152], [185], [180], [196], [142], [190]])
	
	def __len__(self):
		return len(self.x_data)
	
	def __getitem__(self, idx):
		x = torch.FloatTensor(self.x_data[idx])
		y = torch.FloatTensor(self.y_data[idx])
		
		return x, y
# 데이터셋 선언
dataset = CustomDataset()  

# 데이터로더 선언. 미니배치를 만들어준다.
dataloader = DataLoader(  
	dataset, 
	batch_size = 2, 
	shuffle = True)  

# multivariate Linear Regression을 위한 모델 생성. + weight 초기화
class MultivariateLinearRegressionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(3, 1)
		nn.init.constant_(self.linear.weight, 0)   # 첫번째 레이어의 weight를 0으로 초기화 시켜준다.
		
	def forward(self, x):
		return self.linear(x)
	
# model 선언
model = MultivariateLinearRegressionModel()
# optimizer (SGD) 선언
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)


# train 
epochs = 1000
for epoch in range(epochs+1):
	for i, samples in enumerate(dataloader):
		x_train, y_train = samples
		
		hypothesis = model(x_train)
		
		cost = F.mse_loss(hypothesis, y_train)
		
		optimizer.zero_grad()
		cost.backward()
		optimizer.step()
		
	if epoch % 100 == 0:
		print("epochs: {}/{}, cost:{}, y_train: {}, hypothesis:{}".format(
			epoch, epochs, cost.item(), y_train, hypothesis))