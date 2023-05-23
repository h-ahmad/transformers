#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:41:25 2023

@author: hussain
"""

import argparse
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
import pandas as pd
from skimage import io
from efficientnet_pytorch import EfficientNet
from torch.nn import Linear
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/', help="Location of dataset")
    parser.add_argument("--datasets", type=list, default = ['mnist', 'mnist_m', 'usps'], choices=['mnist', 'mnist_m', 'svhn', 'usps'], help="List of datasets")
    parser.add_argument('--test_dataset', type=str, default = 'svhn', help = 'Name of dataset not included in the training datasets.')
    parser.add_argument('--client_per_dataset', type=int, default = 3, choices=[1, 3], help = 'Each dataset has its own number of clients')
    parser.add_argument('--num_classes', type = int, default = 10, choices = [2, 10], help = 'Number of classes in dataset')
    parser.add_argument('--model', type=str, default = 'vit_tiny', choices=['vit_small', 'vit_tiny', 'vit_base', 'efficient', 'cnn2', 'lenet1', 'resnet18', 'custom'], help='Choose model')
    parser.add_argument('--pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument('--resize', type = int, default = 224, choices = [0, 224], help = 'Resize to 224 for transformer, -1 otherwise.')
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Type of optimizer")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--learning_rate", default=0.01, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply. E.g., 0 for SGD and 0.05 for AdamW")
    parser.add_argument("--batch_size", default=128, type=int,  help="Local batch size for training")
    parser.add_argument("--epoch", default=1, type=int, help="Local training epochs in FL")
    parser.add_argument("--communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0,1,2")
    args = parser.parse_args()
    
    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")
    if args.model in 'cnn2':
        model = CNN2Model()
    if args.model in 'lenet1':
        model = LeNetModel()
    if args.model == 'custom':
        model = Net()
    if args.model == 'resnet18':
        if args.pretrained == True:
            model = torch_models.resnet18(pretrained=True)
        else:
            model = torch_models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
    if args.model == 'efficient':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        model._fc = nn.Linear(model._fc.weight.shape[1], args.num_classes)
    if args.model == 'vit_tiny':
        from timm.models.vision_transformer import vit_tiny_patch16_224
        model = vit_tiny_patch16_224(pretrained=args.pretrained)
        model.head = Linear(model.head.weight.shape[1], args.num_classes)
    if args.model == 'vit_small':
        from timm.models.vision_transformer import vit_small_patch16_224
        model = vit_small_patch16_224(pretrained=args.pretrained)
        model.head = Linear(model.head.weight.shape[1], args.num_classes)
    if args.model == 'vit_base':
        from timm.models.vision_transformer import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained=args.pretrained)
        model.head = Linear(model.head.weight.shape[1], args.num_classes)
    model.to(args.device)
    
    model_avg = deepcopy(model).cpu()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_columns = []
    test_columns = []
    model_all = {}
    # optimizer_all = {}
    best_global_val_accuracy = {}
    best_global_test_accuracy = {}
    best_global_val_loss = {}
    best_global_test_loss = {}
    common_index = 0
    for index, client in enumerate(args.datasets):
        for j in range(args.client_per_dataset):
            val_columns.append(client+str(j))
            test_columns.append(client+str(j))
            model_all[common_index] = deepcopy(model).cpu()
            # optimizer_all[index] = optimizer
            best_global_val_accuracy[common_index] = 0
            best_global_test_accuracy[common_index] = 0
            best_global_test_loss[common_index] = 0
            best_global_val_loss[common_index] = 99
            common_index = common_index + 1
    val_csv_file = pd.DataFrame(columns=val_columns)
    test_csv_file = pd.DataFrame(columns = test_columns)      
    for comm_round in range(args.communication_rounds):
        common_index = 0
        for index, client in enumerate(args.datasets):
            for j in range(args.client_per_dataset):
                # print('Training client ==> ', (common_index+1), ' having data ==>', client+str(j),' for communication round ==> ', (comm_round+1))
                # train_loader, _ = load_data(args, client, phase = 'train', client_index = j)
                trainset = FLDataset(args, client, phase = 'train', client_index = j)
                train_loader = DataLoader(trainset, batch_size = args.batch_size, num_workers=args.num_workers)
                model = model_all[common_index].to(args.device)
                if args.model in 'resnet':
                    model.eval()
                else:
                    model.train()
                
                if args.optimizer_type == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5, weight_decay=args.weight_decay)
                elif args.optimizer_type == 'adamw':
                    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)
                
                for epoch in range(args.epoch):
                    correct = 0
                    avg_loss = 0
                    for step, batch in enumerate(train_loader):
                        batch = tuple(t.to(args.device) for t in batch)
                        x, y = batch
                        
                        # import matplotlib.pyplot as plt
                        # plt.imshow(x[0].permute(1,2,0).cpu())

                        optimizer.zero_grad()
                        prediction = model(x)
                        pred = prediction.argmax(dim=1, keepdim=True)
                        correct += pred.eq(y.view_as(pred)).sum().item()
                        loss = loss_fn(prediction.view(-1, args.num_classes), y.view(-1))
                        loss.backward()
                        optimizer.step()
                        avg_loss += loss.item()                
                    avg_loss = avg_loss/len(train_loader.dataset)
                    accuracy = 100*(correct/len(train_loader.dataset))
                    print('Round: {}'.format(comm_round+1), '    epoch: {}'.format(epoch), '    loss: {:.4f}'.format(avg_loss), '   accuracy: {:.4f}'.format(accuracy))
        
                model.to('cpu')
                common_index = common_index + 1
            
        # average model
        global_model = averaging(args, model_avg, model_all) 
        
        print('============= Validation of communication round: ', comm_round+1, '============')
        comm_index = 0
        val_loss = {}
        val_accuracy = {}            
        test_loss = {}
        test_accuracy = {}
        record_val_acc = {}
        record_test_acc = {}
        # _, test_loader = load_data(args, args.test_dataset, phase = 'test')
        testset = FLDataset(args, args.test_dataset, phase = 'test')
        test_loader = DataLoader(testset, batch_size = args.batch_size, num_workers=args.num_workers)
        for index, dataset in enumerate(args.datasets):
            # _, val_loader = load_data(args, dataset, phase = 'val')
            valset = FLDataset(args, dataset, phase = 'val')
            val_loader = DataLoader(valset, batch_size = args.batch_size, num_workers=args.num_workers)
            for client in range(args.client_per_dataset):
                model = model_all[comm_index]
                model.to(args.device)                   
                val_loss[comm_index], val_accuracy[comm_index] = validation(args, model, loss_fn, val_loader)
                if best_global_val_accuracy[comm_index] < val_accuracy[comm_index]:
                    best_global_val_accuracy[comm_index] = val_accuracy[comm_index]
                    best_global_val_loss[comm_index] = val_loss[comm_index]
                    
                    test_loss[comm_index], test_accuracy[comm_index] = validation(args, model, loss_fn, test_loader)
                    if best_global_test_accuracy[comm_index] < test_accuracy[comm_index]:
                        best_global_test_accuracy[comm_index] = test_accuracy[comm_index]   
                        # best_global_test_loss[comm_index] = test_loss[comm_index]
                                        
                record_val_acc[dataset+str(client)] = val_accuracy[comm_index]
                record_test_acc[dataset+str(client)] = best_global_test_accuracy[comm_index]                    
                print('validation accuracy: {:.4f}'.format(val_accuracy[comm_index]), '   dataset: ', dataset, '  client: ', client, ' model: ', comm_index)
                comm_index = comm_index + 1
                model.train()
                model.cpu()    
                
        os.makedirs('results_'+args.test_dataset+'_'+args.model, exist_ok=True)
        output_path = 'results_'+args.test_dataset+'_'+args.model
        
        val_csv_file = val_csv_file.append(record_val_acc, ignore_index=True)
        val_csv_file.to_csv(os.path.join(output_path, 'val_acc.csv'))
        
        test_csv_file = test_csv_file.append(record_test_acc, ignore_index=True)
        test_csv_file.to_csv(os.path.join(output_path, 'test_acc.csv'))
        
        # best_global_test_loss = [val for val in best_global_test_loss.values() if not val == []]
        best_test_accuracy = [val for val in best_global_test_accuracy.values() if not val == []]
        # best_global_test_loss = np.asarray(best_global_test_loss).mean()
        best_test_accuracy = np.asarray(best_test_accuracy).mean()
        print('comm_round: ', comm_round, ', best test accuracy: {:.4f}'.format(best_test_accuracy))
        columns = ['accuracy']
        data = pd.DataFrame(columns = columns)
        data = data.append({
        'accuracy': best_test_accuracy,
        },ignore_index=True)
        data.to_csv(os.path.join(output_path, 'best_accuracy.csv'))

    print('<===== End Training/Testing!======>')
        

if __name__ == "__main__":
    main()