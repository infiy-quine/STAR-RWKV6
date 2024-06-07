import os
assert 'KERAS_BACKEND' in os.environ

if os.environ['KERAS_BACKEND'] == 'jax':
    from .jax_rwkv_kernel import RWKVKernelOperator
elif os.environ['KERAS_BACKEND'] == 'torch':
    from .torch_rwkv_kernel import RWKVKernelOperator
else:
    from .ops_rwkv_kernal import RWKVKernelOperator

from .ops_rwkv_kernal import RWKVKernelOperator as OPSKernelOperator


"""
新增三个参数
return_state 布尔类型 是否返回最终的state,如果想自定义init_state也需要启用这个开关

init_state
    当init_state省缺时，则使用全零初始化BatchSize维度上的状态。
    形状: (state_kinds,num_heads,head_size, head_size)， 其中state_kinds为小于等于Batch_Size的正整数
    精度: 在r为fp16时 init_state为fp32 其余时候类型与r相同


state_map
    形状: (Batch_Size,)
    精度: int64, list[int]
    这个数组定义了state到r上每个Batch维度切片间的映射关系
    取值范围: [0, state_kinds)

返回:
    output, output_state 

def __call__(self,r, k, v, w, u, return_state=False, init_state=None, state_map=None):




"""