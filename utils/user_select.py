import random
class User():
    def __init__(self, di, li,ci,ti,con, index):
        self.di = di #数据量
        self.li = li #计算能力
        self.ci = ci #带宽
        self.ti = ti #传输时间
        self.con = con #贡献
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


def userchoose(b_all, t_all,num_item,num_user):
    idxs_users = []
    users = []
    cals = []
    cons = []
    cs = []
    times = []
    alpha = 0.2
    beta = 2.0
    gama = 2.0
    k = 0.5
    f_k = 0.7
    w = 4


    num_item = one_maxmin(num_item,num_user)

    for i in range(num_user):
        cal = random.randint(3,7)
        cals.append(cal)

    for i in range(num_user):
        con = num_item[i]/((1+alpha)**(-1*cals[i]))
        cons.append(con)

    for i in range(num_user):
        cs.append(2**num_item[i]*2)


    for i in range(num_user):
        time = (k*w)/cs[i]
        times.append(time)


    for i in range(num_user):
        user = User(num_item[i],cals[i],cs[i],times[i],cons[i],i)
        users.append(user)
    users.sort(key=lambda x:x.con,reverse=True)


    C = 0
    T = 0

    for i in range(num_user):
        C = C + users[i].ci
    for i in range(num_user):
        users[i].ci = users[i].ci / C * beta * b_all

    for i in range(num_user):
        T = T + users[i].ti
    for i in range(num_user):
        users[i].ti = users[i].ti/T * gama * t_all

    for i in range(num_user):
        if b_all > 0 and t_all > 0:
            if users[i].ci < b_all and users[i].ti < t_all:
                idxs_users.append(users[i])
                b_all = b_all - users[i].ci
                t_all = t_all - users[i].ti
            else:
                break
    return set(idxs_users)