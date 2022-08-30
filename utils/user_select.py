import random
import math

import numpy as np


class User():
    def __init__(self, di, li, ci, ti,p1, p,index):
        self.di = di  # 数据量
        self.li = li  # 计算能力
        self.ci = ci  # 带宽
        self.ti = ti  # 完成时间（传输+计算）
        self.p1 = p1  # 完成概率
        self.p = p  # 加权概率
        self.index = index


def one_maxmin(num_item, num_user):
    max = 0
    min = 60000
    for num in num_item:
        if num < min:
            min = num
        if num > max:
            max = num
    for i in range(num_user):
        num_item[i] = ((num_item[i] - min) / (max - min) + 0.1)
    return num_item


def userchoose(b_all, t_all, num_user):
    idxs_users = []
    users = []
    alpha = 0.4
    beta = 0.17
    be = 1.5  # 调节选中用户个数的参数
    gama = 1.5  # 调节选中用户个数的参数
    theta = 0.95  # 调节用户被选中后掉线的参数
    k = 0.5
    f_k = 0.7
    w = 4

    num_item = [300, 300, 300, 300, 300, 400, 400, 400, 400, 400,  500, 500, 500, 500, 500, 600, 600, 600, 600, 600]

    # num_item = one_maxmin(num_item[:num_user], num_user)
    num_item = (np.array( num_item )/ 300).tolist()

    cal = [1,1,2,2,2,1,1,2,2,2,1,1,2,2,2,1,1,2,2,2,1,1,2,2,2]
    p1 = [1]*20
    # 初始化每个用户的参数
    for i in range(num_user):
        # cal = random.randint(3, 7)
        p2 = num_item[i] / ((1 + beta) ** (-1 * cal[i])) # 数据量以及计算能力贡献
        bandwidth = min(i *0.1+0.1,1)
        p = alpha * p1[i] + (1-alpha)*p2
        time = round(num_item[i] / 100 + 0.18*num_item[i]/cal[i],3)
        user = User(num_item[i], cal[i], bandwidth, time,p1[i], p ,i)  # ( di, li,ci,ti,p2, index):
        users.append(user)
    users.sort(key=lambda x: x.p, reverse=True)

    C = 0
    T = 0

    # # users[i].ci 归一化
    # for i in range(num_user):
    #     C = C + users[i].ci
    # for i in range(num_user):
    #     users[i].ci = users[i].ci / C * be * b_all
    #
    # # users[i].ti 归一化
    # for i in range(num_user):
    #     # print('users[i].ti',users[i].ti)
    #     T = T + users[i].ti
    # for i in range(num_user):
    #     users[i].ti = users[i].ti / T * gama * t_all

    j = 5
    for i in range(num_user):
        if j > 0:
            idxs_users.append(users[i])
            # print( '选中的：', 'index: {:}, 数据量{:},计算能力{:},带宽{:.1f},传输时间{:.2f}, 概率{:.2f}'.format(users[i].index, users[i].di, users[i].li,users[i].ci, users[i].ti, users[i].p))
            j = j -1                    # t_all = t_all - users[i].ti
    return set(idxs_users)

