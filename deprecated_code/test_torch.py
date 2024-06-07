import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
from rwkv6_model import RWKV6
bz,seq_len,hd_sz = 4,16,512
x = torch.randint(low=0,high=10000,size=(bz,seq_len),dtype=torch.int64)
head_size = 64
rwkv = RWKV6(num_layers=4,vocabulary_size=65536,hidden_size=hd_sz,sequence_length=1024,decomposer_size=hd_sz // 2,head_size=head_size,expand_size=hd_sz * 3)
o = rwkv(x)
print(rwkv.trainable_weights)