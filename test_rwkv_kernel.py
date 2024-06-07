import os 
os.environ['KERAS_BACKEND'] = 'torch'
import numpy as np
import torch
from rwkv_kernel.OpsKernel import RWKVKernelOperator as OpsKernel


from tqdm import tqdm
B  = 4
seq_len = 128
CHANNELS = 512
head_size = 64

kernel1 = OpsKernel(head_size=head_size,max_sequence_length=seq_len)

def RWKV_Kernel(B, T, C, H, r, k, v, w, u):
    D = C // H

    w = torch.reshape(w, [B,T,H,D,1])
    k = torch.reshape(k, [B,T,H,D,1])

    v = torch.reshape(v, [B,T,H,1,D])

    r = torch.reshape(r,[B,T,H,1,D])
    u = torch.reshape(u, [1,H,D,1])

    s = torch.zeros(B,H,D,D,device=u.device).to(u.dtype)

    w = torch.exp(-torch.exp(w))

    kv = k @ v
    #print('kv',kv.shape)
    #print('w',w.shape)

    y_list = []
    for i in range(T):
        kv_t = kv[:,i,:,:,:]
        w_t =  w[:,i,:,:,:]
        r_t = r[:,i,:,:,:]
        y_t = r_t @ (u * kv_t + s)

        y_t = torch.reshape(y_t,(B,C))
        s = kv_t +w_t * s
        y_list.append(y_t)
    x = torch.stack(y_list,axis=1)
    
    #print('sad')
    #print(x.shape)
    
    x = torch.reshape(x,(B,T,C))
    return x

pass_time = 0
for i in tqdm(range(1000)):
    r = np.random.uniform(size=(B,seq_len,CHANNELS),low=100.,high=100.)
    r = torch.tensor(r,dtype=torch.float32).cuda(0)

    k = np.random.uniform(size=(B,seq_len,CHANNELS),low=100.,high=100.)
    k = torch.tensor(k,dtype=torch.float32).cuda(0)

    v = np.random.uniform(size=(B,seq_len,CHANNELS),low=100.,high=100.)
    v = torch.tensor(v,dtype=torch.float32).cuda(0)

    u = np.random.uniform(size=(CHANNELS,),low=100.,high=100.)
    u = torch.tensor(u,dtype=torch.float32).cuda(0)

    w = np.random.uniform(low=-5,high=-0.0001,size=(B,seq_len,CHANNELS)) 
    w = torch.tensor(w,dtype=torch.float32).cuda(0)



    o1 = kernel1(r, k, v, w, u)
    o1 = torch.reshape(o1,(B,seq_len,CHANNELS))

    o2 = RWKV_Kernel(B,seq_len,CHANNELS,CHANNELS//head_size,r, k, v, w, u)
    o2 = torch.reshape(o2,(B,seq_len,CHANNELS))

    #print((o1.cpu() - o2.cpu()).mean())
    pass_time += float((o1.cpu() - o2.cpu()).mean() <=2*1e-4)
print('pass: ',pass_time)