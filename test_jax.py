import os 

os.environ['KERAS_BACKEND'] = 'jax'
import jax.numpy as jnp
from rwkv_kernel.jax_rwkv_kernel import RWKVKernelOperator

seq_len = 512
head_size = 64
channels = 512
bz = 4

inputs_r = jnp.ones(shape=(bz,seq_len,channels))
inputs_k = jnp.ones(shape=(bz,seq_len,channels))
inputs_v = jnp.ones(shape=(bz,seq_len,channels))
inputs_w = jnp.zeros(shape=(bz,seq_len,channels)) - 4
inputs_u = jnp.ones(shape=(channels,))
inputs_s = jnp.ones(shape=(bz,seq_len,channels))

kernel = RWKVKernelOperator(head_size,seq_len)

y = kernel(inputs_r,inputs_k,inputs_v,inputs_w,inputs_u)
print(y)