#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:26:17 2023

@author: hussain
"""

import numpy as np
import torch
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
import argparse
from torch.utils.data.dataloader import DataLoader
import os
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default='32', help="batch size")
    parser.add_argument("--num_workers", type=int, default='4', help="Number of workers")
    parser.add_argument("--channels", type=int, default='3', help="Number of image channels")
    args = parser.parse_args()
    
    # cifar10(args)
    
def cifar10(args):
    data = np.load('./myTr/data/cifar10.npy', allow_pickle=True)
    split = 'split_1'
    print(data.item().keys())
    data = data.item()
    print(data[split].keys())
    # data, target
    print(data[split]['data'].keys())
    print(data[split]['target'].keys())
    # data (shape)
    print('train_1 shape: ', data[split]['data']['train_1'].shape)  # train_1, train_2, train_3, train_4, train_5
    print('length target list for train_1: ', len(data[split]['target']['train_1']))
    target = torch.tensor(data[split]['target']['train_1'])
    print('target shape: ', target.shape)
    labels, counts = torch.unique(target, sorted = True, return_counts = True)
    print('lable: ', labels, ', count: ', counts)

def load_data():
    second_index = 1
    data_file = os.path.join(os.path.join('./data', 'mnist'), 'mnist_train_test.pkl')
    with open(data_file, 'rb') as file:
            data_store = pickle.load(file)
    xTrain, yTrain, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']    
    print(xTrain.shape, yTrain.shape)
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (xTrain.astype(np.float32), yTrain.astype(np.int_), 
                                                      xTest.astype(np.float32), yTest.astype(np.int_))) 
    yTrain = yTrain.type(torch.LongTensor)
    yTest = yTest.type(torch.LongTensor)
    train_x = xTrain[second_index]
    train_y = yTrain[second_index]
    test_x = xTest[second_index]
    test_y = yTest[second_index]
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=128)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=128)
    

if __name__ == "__main__":
    # main()
    load_data()
    
