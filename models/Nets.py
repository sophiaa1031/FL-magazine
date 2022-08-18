#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_FEMNIST(nn.Module):
    def __init__(self, args):
        super(CNN_FEMNIST, self).__init__()
        self.conv1x = nn.Conv2d(1, 4, 5)
        self.poolx = nn.MaxPool2d(2, 2)
        self.conv2x = nn.Conv2d(4, 12, 5)
        self.fc1x = nn.Linear(12 * 4 * 4, 120)
        self.fc2x = nn.Linear(120, 100)

        self.conv1y = nn.Conv2d(1, 4, 5)
        self.pooly = nn.MaxPool2d(2, 2)
        self.conv2y = nn.Conv2d(4, 12, 5)
        self.fc1y = nn.Linear(12 * 4 * 4, 120)
        self.fc2y = nn.Linear(120, 100)

        self.fc3 = nn.Linear(100, args.num_classes)

        self.weight_keys = [['fc1x.weight', 'fc1x.bias'],
                            ['fc2x.weight', 'fc2x.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2x.weight', 'conv2x.bias'],
                            ['conv1x.weight', 'conv1x.bias'],
                            ['fc1y.weight', 'fc1y.bias'],
                            ['fc2y.weight', 'fc2y.bias'],
                            ['conv2y.weight', 'conv2y.bias'],
                            ['conv1y.weight', 'conv1y.bias']
                            ]

    def forward(self, x):
        y = x.clone()
        x = self.poolx(F.relu(self.conv1x(x)))
        x = self.poolx(F.relu(self.conv2x(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1x(x))
        x = F.relu(self.fc2x(x))

        y = self.pooly(F.relu(self.conv1y(y)))
        y = self.pooly(F.relu(self.conv2y(y)))
        y = y.view(-1, 12 * 4 * 4)
        y = F.relu(self.fc1y(y))
        y = F.relu(self.fc2y(y))

        z = F.cosine_similarity(x, y)
        z = torch.mean(z)
        x = torch.cat((x, y), 1)

        x = self.fc3(x)
        return x