import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import torch
from keras.src.layers import GroupNormalization,LayerNormalization
import numpy as np
from torch.nn import GroupNorm
from keras.src import ops
inputs = np.random.uniform(low=-100,high=100,size=(768,2048))
inputs = torch.tensor(inputs,dtype=torch.float32)
from keras.src.initializers import RandomUniform
init = RandomUniform(seed=1234,minval=-4,maxval=4)
group_norm1 = GroupNormalization(groups=32,center=True,scale=True,epsilon=64*1e-5,beta_initializer=init,gamma_initializer=init)
group_norm2 = GroupNorm(num_groups=32,num_channels=2048,affine=True,eps=64*1e-5)
group_norm3 = LayerNormalization(epsilon=64*1e-5,center=False,scale=False)


group_norm1(inputs)

print(group_norm1.gamma.shape)
print(group_norm2.weight.shape)

gamma = group_norm1.gamma.numpy()
beta  = group_norm1.beta.numpy()

group_norm2.weight = torch.nn.Parameter(torch.tensor(gamma,dtype=torch.float32).to(group_norm2.weight.device))
group_norm2.bias = torch.nn.Parameter(torch.tensor(beta,dtype=torch.float32).to(group_norm2.weight.device))


o3 = group_norm3(ops.convert_to_tensor(inputs.reshape(768,32,64))).reshape((768,2048))
o3 = o3 * ops.convert_to_tensor(gamma) + ops.convert_to_tensor(beta)


o1 = group_norm1(inputs).detach().cpu().numpy()

o2 = group_norm2(inputs).detach().cpu().numpy()

o3 = o3.detach().cpu().numpy()


print(np.abs(o1 -o2).mean())
print(np.abs(o2 -o3).mean())

