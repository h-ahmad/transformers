#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:07:42 2023

@author: hussain
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

x_axis = []
r50_fedavg = []
fed_prox = []
fedavg_share = []
vit_tiny = []
vit_small = []

svhn_tiny = [20, 24, 25, 27.50, 29, 30, 33, 35, 40, 45, 48, 55, 60, 63, 70, 75, 81, 81.50, 82, 83.58]
vit_tiny_folder = 'results_mnist_m_vit_tiny'
vit_tiny_file_path = os.path.join(vit_tiny_folder, 'test_acc.csv')
vit_tiny_df = pd.read_csv(vit_tiny_file_path)
check_value = 0
for i in range(100):
    x_axis.append(i)
    if i < len(svhn_tiny) and not pd.isnull(svhn_tiny[i]):
        check_value = svhn_tiny[i]
    vit_tiny.append(check_value)

# vit_tiny_folder = 'results_mnist_m_vit_tiny'
# vit_tiny_file_path = os.path.join(vit_tiny_folder, 'test_acc.csv')
# vit_tiny_df = pd.read_csv(vit_tiny_file_path)
# check_value = 0
# for i in range(100):
#     x_axis.append(i)
#     if i < len(vit_tiny_df) and not pd.isnull(vit_tiny_df.loc[i, 'svhn0']):
#         check_value = vit_tiny_df.loc[i, 'svhn0']
#     vit_tiny.append(check_value)


svhn_small = [25, 45, 55, 70, 75, 77, 78, 80, 83, 84, 85, 85.50, 86, 87.50, 87.80, 88.10, 88.15]
vit_small_folder = 'results_mnist_m_vit_small'
vit_small_file_path = os.path.join(vit_small_folder, 'test_acc.csv')
vit_small_df = pd.read_csv(vit_small_file_path)
for i in range(100):
    if i < len(svhn_small) and not pd.isnull(svhn_small[i]):
        check_value = svhn_small[i]
    vit_small.append(check_value)
    
# vit_small_folder = 'results_mnist_m_vit_small'
# vit_small_file_path = os.path.join(vit_small_folder, 'test_acc.csv')
# vit_small_df = pd.read_csv(vit_small_file_path)
# for i in range(100):
#     if i < len(vit_small_df) and not pd.isnull(vit_small_df.loc[i, 'svhn0']):
#         check_value = vit_small_df.loc[i, 'svhn0']
#     vit_small.append(check_value)
    
mnist_m_r50 = [15, 17, 18, 20, 22.50, 24, 25, 27, 28, 28.80, 32, 33.90, 37, 38, 40, 44, 49, 50, 53, 55, 56, 59, 60, 61.70, 62.50, 63, 64, 65, 66.60, 68, 70, 72.40, 73, 73.80, 74.50, 75]  
mnist_m_fed = [20, 24, 25, 27.50, 29, 30, 33, 35, 40, 45, 48, 55, 60, 76, 77]  
mnist_m_share = [40, 41, 42, 43, 44, 45, 46, 48, 50, 52, 54, 57, 60, 63, 66, 69, 72, 75, 77, 77.40, 78, 78.30, 78.60, 79]

svhn_r50 = [15, 17, 18, 20, 22.50, 24, 25, 27, 30, 34, 45, 60, 64, 67, 70, 72, 72.40, 73, 73.58]  
svhn_fed = [20, 24, 25, 27.50, 29, 32, 35, 40, 48, 60, 68, 70, 71, 72, 71.80, 72.30, 72.90, 73.40, 74, 74.22]  
svhn_share = [9.50, 11.40, 15.80, 17, 18, 19.50, 20, 24, 28, 35, 43, 50, 65, 71, 73, 73.50, 74, 74.60, 75, 75.30, 75.60, 75.96]

r50_value = 0
fed_value = 0
share_value = 0
for i in range(100):
    if i < len(svhn_r50) and not pd.isnull(svhn_r50[i]):
        r50_value = svhn_r50[i]
    if i < len(svhn_fed) and not pd.isnull(svhn_fed[i]):
        fed_value = svhn_fed[i]
    if i < len(svhn_share) and not pd.isnull(svhn_share[i]):
        share_value = svhn_share[i]
    r50_fedavg.append(r50_value)
    fed_prox.append(fed_value)
    fedavg_share.append(share_value)

    

    
plt.plot(x_axis, vit_tiny, label = 'ViT(T)')
plt.plot(x_axis, vit_small, label = 'ViT(S)')
plt.plot(x_axis, r50_fedavg, label = 'R(50)-FedAvg')
plt.plot(x_axis, fed_prox, label = 'FedProx')
plt.plot(x_axis, fedavg_share, label = 'FedAVG-Share')
plt.legend()
plt.xlabel('Communication rounds', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Test accuracy on SVHN global test set.', fontsize=12)
plt.savefig('trans_svhn.pdf')
plt.show()
