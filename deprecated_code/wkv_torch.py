import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


B = 1
T = 16
C = 8
H = 1
D = 8
#初始化
w1 = torch.zeros(size=[B,T,H,D,1]).float() - 100
w1.requires_grad = True
k = torch.zeros(size=[B,T,H,D,1]).float() + 2
k.requires_grad = True
v = torch.zeros(size=[B,T,H,1,D]).float() + 2
v.requires_grad = True
r = torch.zeros(size=[B,T,H,1,D]).float() + 2
r.requires_grad = True
u = torch.zeros(size=[1,H,D,1]).float() + 2
u.requires_grad = True
y_list = []
s = torch.zeros(B,H,D,D).float()
s.requires_grad = True
#正反向传播
w = torch.exp(-torch.exp(w1))
kv = k @ v

mask = torch.randint(low=0, high=1, size=(B,T)).bool()#掩码矩阵


for i in range(T):
    kv_t = kv[:,i,:,:,:]
    w_t =  w[:,i,:,:,:]
    r_t = r[:,i,:,:,:]

    m_t = 1 - mask[:,i].to(kv_t.dtype).reshape((B,1,1,1)) #将掩码转换成float32矩阵,并相减
    s = s * m_t#此时的m_t为0时候会清空梯度，（m_t[B]为0 对应mask[B,i]为True）


    y_t = r_t @ (u * kv_t + s)

    

    print(y_t.shape)
    print(r.shape)
    print(u.shape)
    print(kv_t.shape)
    print(s.shape)
    y_t = torch.reshape(y_t,(B,C))
    s = kv_t +w_t * s
    y_list.append(y_t)
x = torch.stack(y_list,axis=1)

x.backward(torch.ones(size=x.shape),retain_graph=True)
#打印结果
#print(w)
#print(k)
#print(v)
#print(r)
#print(u)
#print(y)
#print(s)
#print(x)
#print(w1.grad)
#print(k.grad)
#print(v.grad)
#print(r.grad)
#print(u.grad)
#print(y.grad)





