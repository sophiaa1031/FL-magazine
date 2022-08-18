#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import xlwt
from torchvision import datasets, transforms
import torch

import warnings
warnings.filterwarnings("ignore")

from utils.user_select import userchoose
from utils.user_select_randomly import userchoose_randomly

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNN_FEMNIST
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':

    # parse args
    num_users = 10
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (
        0.3081,))])  # ToTensor()转换一个PIL库的图片或者numpy的数组为tensor张量类型；转换从[0,255]->[0,1]
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=trans_mnist)  # Normalize通过平均值和标准差来标准化一个tensor图像，公式为：
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            num_items = [1000, 1500, 2000, 2500, 3000, 3000, 3500, 4000, 4500, 5000]
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            num_items = [300, 300, 600, 600, 900, 900, 1200, 1200, 1500, 1500]
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            num_items = [500, 1000, 1500, 2000, 2000, 25000, 3000, 3500, 4000, 5000]
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            num_items = [300, 300, 600, 600, 900, 900, 1200, 1200, 1500, 1500]
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fminist':
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True,
                                              transform=transforms.ToTensor())
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True,
                                             transform=transforms.ToTensor())
        if args.iid:
            num_items = [500, 1000, 1500, 2000, 2000, 25000, 3000, 3500, 4000, 5000]
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            num_items = [300, 300, 600, 600, 900, 900, 1200, 1200, 1500, 1500]
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_train = []
    cv_loss, cv_acc = [], []
    trans_time_round = []
    dura_times = [0,0,0,0,0,0,0,0,0,0]

    trans_time = 0
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):

        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if args.user_select is True:
            idxs_users = userchoose(20, 20, num_items, 10)
        else:
            # m = max(int(args.frac * args.num_users), 1)
            idxs_users = userchoose_randomly(20,num_items,10,3)
        trans_time_per_round = 0
        for user in idxs_users:
            # print(user.index)
            trans_time_per_round = trans_time_per_round + user.ti
            trans_time = trans_time + user.ti
        trans_time_round.append(trans_time_per_round)


        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx.index], epc=idx.li)
            w, loss, dura_time = local.train(net=copy.deepcopy(net_glob).to(args.device))
            dura_times[idx.index] = dura_times[idx.index] + dura_time
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_train.append(float(str("{:.2f}".format(acc_test.item()))))
        print('Round {:3d}, Average loss {:.4f}'.format(iter, loss_avg))
        print('Round {:3d}, global_test loss {:.4f}'.format(iter, loss_test))
        loss_train.append(float(str("{:.2f}".format(loss_avg))))

    print('loss is: ',loss_train)
    print("Transmission time：{:.2f}".format((trans_time)))
    print('acc_train is: ',acc_train)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print(dura_times)
    # print('loss is: ', (loss_sum / args.loop).tolist())
    # print('accuracy is ', (acc_sum / args.loop).tolist())
