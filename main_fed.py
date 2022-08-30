#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math

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
import random


if __name__ == '__main__':

    # parse args
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
            num_items = [1000, 1500, 2000, 2500, 3000, 3000, 3500, 4000, 4500, 5000]  #每个用户的数据量大小，注意相加之和不能超过总数据量
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            num_items = [300, 300,300,300,300,400, 400, 400, 400, 400, 500, 500, 500,  500, 500, 600,600, 600,600,600]
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            num_items = [1000, 1500, 2000, 2500, 3000, 3000, 3500, 4000, 4500, 5000]
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            num_items = [300, 300, 600, 600, 900, 900, 1200, 1200, 1500, 1500]
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True,
                                              transform=transforms.ToTensor())
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True,
                                             transform=transforms.ToTensor())
        if args.iid:
            num_items = 300 * np.array([i for i in range(1,args.num_users+1)])
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            num_items = [300, 300,300,300,300, 400, 400, 400, 400, 400,  500, 500, 500,  500, 500, 600,600, 600,600,600]
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    # print('dict_users: ', len(dict_users[0]),len(dict_users[1]),len(dict_users[2]),len(dict_users[3]),len(dict_users[4]))
    # print('dict_users: ', len(dict_users[0]),len(dict_users[1]),len(dict_users[2]),len(dict_users[3]),len(dict_users[4]),len(dict_users[5]),len(dict_users[6]),len(dict_users[7]),len(dict_users[8]),len(dict_users[9]))
    cal = [1,1,2,2,2,1,1,2,2,2,1,1,2,2,2,1,1,2,2,2,1,1,2,2,2]

    loss_sum = np.array(0)
    latency_sum = np.array(0)
    time_window = 0.5
    for loop_num in range(args.loop):
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
        # print(net_glob)
        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        acc_record = []
        cv_loss, cv_acc = [], []
        trans_time_round = []
        dura_times = [0] * args.num_users

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
                idxs_users = userchoose(20, 2, args.num_users)
            trans_time_per_round = 0

            trans_time_per_round_max = 0
            for idx in idxs_users:
                if args.user_select is True and ((idx.p1 > random.random()) or (trans_time_per_round_max == 0)) or args.user_select is False and (iter % 5 != 0 or  trans_time_per_round_max != 0):
                    print(args.user_select,iter)
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx.index], epc=idx.li)
                    w, loss, dura_time = local.train(net=copy.deepcopy(net_glob).to(args.device)) #dura_time是实际 LocalUpdate 的计算时间
                    # print('dura_time:',dura_time)
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                    time_each_client = round(num_items[idx.index] / 60000 + dura_time / cal[idx.index], 3)
                    # 计算latency
                    if time_each_client < time_window:
                        trans_time_per_round_max = min(max(trans_time_per_round_max, time_each_client),time_window)
                    else:
                        trans_time_per_round_max = time_window
                else:
                    trans_time_per_round_max = time_window
            if len(trans_time_round) == 0:
                trans_time_round.append(trans_time_per_round_max)
            else:
                trans_time_round.append(round(trans_time_per_round_max+trans_time_round[-1],3))
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            # print('Round {:3d}, loss {:.4f}'.format(iter, loss_avg))
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            loss_train.append(float(str("{:.2f}".format(loss_avg))))
            acc_record.append(float(str("{:.2f}".format(acc_test.item()))))
            print('Round {:3d}, acc_test {:.4f}'.format(iter, (acc_test.item())))
        loss_sum = loss_sum + np.array(loss_train)
        latency_sum = latency_sum + np.array(trans_time_round)
        print('acc_record is: ', acc_record)

        # testing
        # net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print('loss is: ', (loss_sum / args.loop).tolist())
    # print('latency is: ', (latency_sum / args.loop).tolist())
