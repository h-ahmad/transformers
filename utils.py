#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:33:30 2023

@author: hussain
"""

import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
import pandas as pd
from skimage import io
import numpy as np
import pickle
from skimage.transform import resize
import random
import torch.nn.functional as F

class CNN2Model(nn.Module):
    def __init__(self):
        super(CNN2Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.activation = nn.ReLU(True)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7,7))   # adaptive pool
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(12) * (7 * 7), out_features=10, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        # print(x.view(-1, self.num_flat_features(x)).shape[1])
        x = self.dense1(x)
        return x
    
class LeNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5)
        self.activation = nn.Tanh()
        # self.pool = nn.AvgPool2d(kernel_size=2)
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear( (12) * (7 * 7), 10)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x        
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.adaptive_max_pool2d(self.conv1(x), 12))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)     
    
# ============================Perfect for MNIST=================================================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
# 
#     def forward(self, x):
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)  
# =============================================================================

def averaging(args, model_avg, model_all):
    print('Model averaging...')
    # model_avg.cpu()
    params = dict(model_avg.named_parameters())    
    for name, param in params.items():
        comm_index = 0
        for index, client in enumerate(args.datasets):
            for j in range(args.client_per_dataset):
                if (index == 0 and j == 0):
                    tmp_param_data = dict(model_all[comm_index].named_parameters())[name].data * (1/(len(args.datasets)*args.client_per_dataset))                    
                else:
                    tmp_param_data = tmp_param_data + dict(model_all[comm_index].named_parameters())[name].data * (1/(len(args.datasets)*args.client_per_dataset))
                comm_index = comm_index + 1
        params[name].data.copy_(tmp_param_data)
    print('Updating clients...')
    common_index = 0
    for index, client in enumerate(args.datasets):
        for j in range(args.client_per_dataset):
            tmp_params = dict(model_all[common_index].named_parameters())
            for name, param in params.items():
                tmp_params[name].data.copy_(param.data)
            common_index = common_index + 1
    return model_avg

# ============================ Evaluation of single model =================================================
# def valid(args, model, loss_fn, data_loader):
#     total_loss = {}
#     total_accuracy = {}
#     model.to(args.device)
#     model.eval()
#     correct = 0
#     loss = 0
#     with torch.no_grad():
#         for step, batch in enumerate(data_loader):
#             batch = tuple(t.to(args.device) for t in batch)
#             x, y = batch                    
#             logits = model(x)
#             loss += loss_fn(logits.view(-1, args.num_classes), y.view(-1))
#             pred = logits.argmax(dim=1, keepdim=True)
#             correct += pred.eq(y.view_as(pred)).sum().item()               
#     total_loss = (loss/len(data_loader.dataset)).cpu()
#     total_accuracy = 100*(correct / len(data_loader.dataset))
#     model.train()
#     return total_loss, total_accuracy
# =============================================================================

def validation(args, model, loss_fn, data_loader):
    total_loss = {}
    total_accuracy = {}
    model.eval()
    correct = 0
    loss = 0
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():                
            logits = model(x)
            loss += loss_fn(logits.view(-1, args.num_classes), y.view(-1))
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()               
    total_loss = (loss/len(data_loader.dataset)).cpu().item()
    total_accuracy = 100*(correct / len(data_loader.dataset))
    return total_loss, total_accuracy

class CustomLoader(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None):
        self.image_names = pd.read_csv(csv_path)
        self.data_path = dataset_path
        self.transform = transform
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.data_path, self.image_names.iloc[index, 0])
        img = io.imread(img_name)
        label = self.image_names.iloc[index, 1]
        label = torch.tensor(label)
        if self.transform:
            img = self.transform(img)
        #sample = {'img':img, 'label':label}
        return img, label 
        
def load_data_folder(args, client, phase, client_index = None):
    csv_path = os.path.join(os.path.join(args.data_path, client), client+'_train.csv')
    dataset_path = os.path.join(os.path.join(args.data_path, client), client+'_train')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomLoader(csv_path, dataset_path, transform)
    trainLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    csv_path = os.path.join(os.path.join(args.data_path, client), client+'_test.csv')
    dataset_path = os.path.join(os.path.join(args.data_path, client), client+'_test')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomLoader(csv_path, dataset_path, transform)
    testLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return trainLoader, testLoader  
    

def load_data(args, client, phase, client_index = None):
    data_file = os.path.join(os.path.join(args.data_path, client), client+'_train_test.pkl')
    with open(data_file, 'rb') as file:
            data_store = pickle.load(file)
    train_x, train_y, test_x, test_y = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test'] 
    train_x, train_y, test_x, test_y = map(torch.tensor, (train_x.astype(np.float32), train_y.astype(np.int_), 
                                                      test_x.astype(np.float32), test_y.astype(np.int_))) 
    train_y = train_y.type(torch.LongTensor)
    test_y = test_y.type(torch.LongTensor)
    if client_index is not None:
        train_x = train_x[client_index]
        train_y = train_y[client_index]
    if (phase == 'train' and train_x.shape[1] == 1):
        train_x = torch.cat((train_x, train_x, train_x), dim=1)
    if (phase == 'val' and test_x.shape[1] == 1):
        test_x = torch.cat((test_x, test_x, test_x), dim=1)
    if (phase == 'test' and test_x.shape[1] == 1):
        test_x = torch.cat((test_x, test_x, test_x), dim=1)  
    trainDs = torch.utils.data.TensorDataset(train_x,train_y)
    testDs = torch.utils.data.TensorDataset(test_x,test_y)
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=args.batch_size)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=args.batch_size)
    return trainLoader, testLoader

class FLDataset(Dataset):
    def __init__(self, args, client, phase, client_index = None):
        super(FLDataset, self).__init__()
        self.phase = phase

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.05, 1.0)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
        
        data_file = os.path.join(os.path.join(args.data_path, client), client+'_train_test.pkl')
        with open(data_file, 'rb') as file:
                data_store = pickle.load(file)
        train_x, train_y, test_x, test_y = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test'] 
        train_x, train_y, test_x, test_y = map(torch.tensor, (train_x.astype(np.float32), train_y.astype(np.int_), 
                                                          test_x.astype(np.float32), test_y.astype(np.int_))) 
        
        if self.phase == 'train':
            self.label = train_y.type(torch.LongTensor)
            if client_index is not None:
                self.data = train_x[client_index]
                self.label = self.label[client_index]
            if (phase == 'train' and self.data.shape[1] == 1):
                self.data = torch.cat((self.data, self.data, self.data), dim=1)
        else:
            self.label = test_y.type(torch.LongTensor)
            if test_x.shape[1] == 1:
                self.data = torch.cat((test_x, test_x, test_x), dim=1)
            else:
                self.data = test_x
        # img, target = self.data, self.label
    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        if self.transform is not None:
            img = self.transform(img)        
        return img,  target
    def __len__(self):
        return len(self.data)

