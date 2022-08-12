import random


class User():
    def __init__(self, li,ti, index):
        self.li = li  # 计算能力
        self.ti = ti  # 传输时间
        self.index = index



def one_maxmin(num_item,num_user):
    max = 0
    min = 60000
    for num in num_item:
        if num < min:
            min = num
        if num > max:
            max = num
    for i in range(num_user):
        num_item[i] = ((num_item[i] - min)/(max - min)+0.1)*2
    return num_item

def userchoose_randomly(t_all,num_item,num_user, num_sel):
    users = []
    idx_users = []
    cals = []
    cs = []
    times = []
    alpha = 0.2
    beta = 1.8
    gama = 1.8
    T = 0
    k = 0.5
    w = 4


    for i in range(num_user):
        cal = random.randint(3, 7)
        cals.append(cal)

    num_item = one_maxmin(num_item, num_user)

    for i in range(num_user):
        cs.append(2**num_item[i]*2)


    for i in range(num_user):
        time = (k * w) / cs[i]
        times.append(time)


    for i in range(num_user):
        user = User(cals[i], times[i],i)
        users.append(user)



    for i in range(num_user):
        T = T + users[i].ti
    for i in range(num_user):
        users[i].ti = users[i].ti/T * gama * t_all

    result = random.sample(range(num_user), num_sel)
    for user in users:
        if user.index in result:
            idx_users.append(user)
    return set(idx_users)
