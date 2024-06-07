import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import jax.numpy as jnp


from rwkv_kernel.jax_rwkv_kernel import RWKVKernelOperator

head_size = 32
seq_len = 768
B = 1
CHANNELS = 512

kernel = RWKVKernelOperator(head_size, seq_len)

r = jnp.ones(shape=(B,seq_len,CHANNELS))

k = jnp.ones(shape=(B,seq_len,CHANNELS))

v = jnp.ones(shape=(B,seq_len,CHANNELS))

u = jnp.ones(shape=(CHANNELS,))

w = np.random.uniform(low=-5,high=-0.0001,size=(B,seq_len,CHANNELS)) 

#test
o2 = kernel(r, k, v, w, u)

print(o2[:,0])



#print(torch.isnan(o1).any(),o1.shape)
#print(torch.isnan(o2).any(),o2.shape)
#print(o1[:,0])
#print(o2[:,0])
#print(torch.abs(o1[:,0] - o2[:,0]).mean(axis=-1))
