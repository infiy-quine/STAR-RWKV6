import os 

os.environ['KERAS_BACKEND'] = 'torch'
from rwkv_kernel.torch_rwkv_kernel import RWKVKernelOperator
import numpy as np
import torch

seq_len = 512
head_size = 64
channels = 512
bz = 4

inputs_r = np.ones(shape=(bz,seq_len,channels))
inputs_r = torch.Tensor(inputs_r,dtype=torch.float32)

inputs_k = np.ones(shape=(bz,seq_len,channels))
inputs_k = torch.Tensor(inputs_k,dtype=torch.float32)

inputs_v = np.ones(shape=(bz,seq_len,channels))
inputs_v = torch.Tensor(inputs_v,dtype=torch.float32)

inputs_w = np.zeros(shape=(bz,seq_len,channels)) - 4
inputs_w = torch.Tensor(inputs_w,dtype=torch.float32)

inputs_u = np.ones(shape=(channels,))
inputs_u = torch.Tensor(inputs_u,dtype=torch.float32)

inputs_s = np.ones(shape=(bz,seq_len,channels))
inputs_s = torch.Tensor(inputs_s,dtype=torch.float32)

kernel = RWKVKernelOperator(head_size,seq_len)

y = kernel(inputs_r,inputs_k,inputs_v,inputs_w,inputs_u)
print(y)