#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import  optim
import torch.nn as nn
import torch.nn.functional as F


def generate_dataset(mySeed):
  np.random.seed(mySeed)  # For reproducibility


  # Coefficients
  theta0, theta1, theta2, theta3 = mySeed * 0.05, mySeed*0.01, mySeed*-0.01, mySeed*0.02

  # Train set
  n_samples_train = 1000

  # Generate predictors
  x = np.random.normal(0, 1, n_samples_train)
  y = np.random.normal(0, 1, n_samples_train)
  z = np.random.exponential(0.5, n_samples_train)

  # Generate targets with added Gaussian noise
  t = theta0 + theta1 * x - theta2 * y + theta3 * z + np.random.normal(0, 2, n_samples_train)

  # Stack inputs and targets
  inputs = np.stack((x, y, z), axis=1)
  targets = t
  train_set = {'inputs':inputs, 'targets':targets}

  # Test set
  n_samples_test = 500

  # Generate predictors
  x = np.random.normal(0, 1, n_samples_test)
  y = np.random.normal(0, 1, n_samples_test)
  z = np.random.exponential(0.5, n_samples_test)

  # Generate targets with added Gaussian noise
  t = theta0 + theta1 * x - theta2 * y + theta3 * z + np.random.normal(0, 2, n_samples_test)

  # Stack inputs and targets
  inputs = np.stack((x, y, z), axis=1)
  targets = t
  test_set = {'inputs':inputs, 'targets':targets}

  return train_set, test_set



class Dataset1(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]




class Double_conv_causal(nn.Module):
    '''(conv => BN => ReLU) * 2, with causal convolutions that preserve input size'''
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(Double_conv_causal, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        # Apply causal padding manually for the first convolution
        x = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0))  # Pad 2 on the left (time dimension), 0 on the right
        x = self.conv1(x)
        # testx
        x = self.bn1(x)
        x = self.relu(x)

        x = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0))
        x = self.conv2(x)
        # testx
        x = self.bn2(x)
        x = self.relu(x)
        return x



class Down_causal(nn.Module):
    ''' Downscaling with maxpool then double conv '''
    def __init__(self, in_ch, out_ch, pooling=True, pooling_kernel_size=2, pooling_stride=2, dilation=1):
        super(Down_causal, self).__init__()
        if pooling:
            self.mpconv = nn.Sequential(
                nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride),
                Double_conv_causal(in_ch, out_ch)
            )
        else:
            self.mpconv = nn.Sequential(
                Double_conv_causal(in_ch, out_ch,dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class Up_causal(nn.Module):
    ''' Upscaling then double conv, modified to maintain causality '''
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2):
        super(Up_causal, self).__init__()

        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0)

        self.conv = Double_conv_causal(2 * in_ch, out_ch)  # Using causal convolution here

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate necessary padding for concatenation to match the earlier layer's feature dimension
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2))  # Symmetric padding for alignment, maintain causality

        x = torch.cat([x1, x2], dim=1)  # Concatenate features from down-path
        x = self.conv(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)


    def forward(self, x):
        x = self.conv(x)
        return x
