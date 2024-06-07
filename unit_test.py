
import os
os.environ['KERAS_BACKEND'] = 'torch'
from keras_torch_transport import TorchToKerasConvertor
from rwkv_v6_demo import RWKV
import torch
import numpy as np
import types
args = types.SimpleNamespace()

args.n_layer = 24
args.n_embd = 2048

args.vocab_size = 65536
args.ctx_len = 2

########################################################################################################
# CUDA Kernel
########################################################################################################

args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change
print(args)
weight_path = r"/home/niconiconi/rwkv_weights/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"

model_params = torch.load(weight_path, map_location="cpu")

rwkv6_torch = RWKV(args)


rwkv6_torch.float().to('cuda:1')
rwkv6_torch.load_state_dict(model_params)

del model_params

convertor = TorchToKerasConvertor(weight_path)
rwkv6_keras = convertor.build_rwkv_from_torch()

bz = 1
inputs = np.random.uniform(size=(bz,args.ctx_len),low=0,high=4000).astype(np.int64)
inputs = torch.tensor(inputs,dtype=torch.int64)

outputs_torch = rwkv6_torch(inputs.cuda(1))
outputs_jax = rwkv6_keras(inputs)
print(inputs.shape)
print(outputs_jax.shape)
def calculate_mae(a,b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return np.abs(a - b).mean()

def calculate_ratio(a,b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return (np.abs(a - b) / (np.abs(a) + 1e-7)).mean()

print("---------- 总体误差如下: ",calculate_mae(outputs_jax,outputs_torch))


hidden_inputs= np.random.uniform(size=(bz,args.ctx_len,args.n_embd),low=0,high=4000).astype(np.float32)
hidden_inputs = torch.tensor(hidden_inputs,dtype=torch.float32)

time_torch = rwkv6_torch.blocks[1].att(hidden_inputs.cuda(1))
time_keras = rwkv6_keras.rwkv_blocks[1].time_mix(hidden_inputs.cuda(0))

print("---------- TimeMix","绝对误差: ",calculate_mae(time_torch,time_keras),"相对误差比例: ",calculate_ratio(time_torch,time_keras))


hidden_inputs= np.random.uniform(size=(bz,args.ctx_len,args.n_embd),low=0,high=4000).astype(np.float32)
hidden_inputs = torch.tensor(hidden_inputs,dtype=torch.float32)

channel_torch = rwkv6_torch.blocks[1].ffn(hidden_inputs.cuda(1))
channel_keras = rwkv6_keras.rwkv_blocks[1].channel_mix(hidden_inputs.cuda(0))

print("---------- ChannelMix误差如下: ",calculate_mae(channel_torch,channel_keras))



hidden_inputs= np.random.uniform(size=(bz,args.ctx_len,args.n_embd),low=0,high=4000).astype(np.float32)
hidden_inputs = torch.tensor(hidden_inputs,dtype=torch.float32)

block_torch = rwkv6_torch.blocks[1](hidden_inputs.cuda(1))
block_keras = rwkv6_keras.rwkv_blocks[1](hidden_inputs.cuda(0))

print("---------- RWKV Block误差如下: ",calculate_mae(block_torch,block_keras))