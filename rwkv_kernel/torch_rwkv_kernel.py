import os
import torch
from torch.utils.cpp_extension import load
from keras import ops
kernel_dir_name = 'torch_kernel'

class RWKVKernelOperator:
    def __init__(self,head_size,max_sequence_length):
        current_dir = os.path.dirname(__file__)
        #current_dir = os.pat
        wkv6_cuda = load(name="wkv6", sources=[os.path.join(current_dir,f"{kernel_dir_name}/wkv6_op.cpp"), os.path.join(current_dir,f"{kernel_dir_name}/wkv6_cuda.cu")],
                        #verbose=True, extra_cuda_cflags=[f"-D_N_={head_size}", f"-D_T_={max_sequence_length}"])
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size}", f"-D_T_={max_sequence_length}"])
        class RWKV_6(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                if not isinstance(u,torch.Tensor):
                    u = u.value
                with torch.no_grad():
                    assert r.dtype ==  k.dtype == v.dtype == w.dtype == u.dtype
                    assert r.dtype in [torch.float32,torch.bfloat16,torch.float16]

                    assert head_size == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u)
                    
                    y_dtype = r.dtype if r.dtype != torch.float16 else torch.float32
                    
                    y = torch.empty((B, T, C), device=r.device, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

                    if r.dtype == torch.float32:
                        wkv6_cuda.forward_fp32(B, T, C, H, r, k, v, w, u, y)
                    elif r.dtype == torch.bfloat16:
                        wkv6_cuda.forward_bf16(B, T, C, H, r, k, v, w, u, y)
                    else:
                        wkv6_cuda.forward_fp16(B, T, C, H, r, k, v, w, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                assert gy.is_cuda
                with torch.no_grad():
                    assert gy.dtype in [torch.bfloat16,torch.float32]
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u = ctx.saved_tensors
                    y_dtype = r.dtype if r.dtype != torch.float16 else torch.float32

                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=y_dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

                    if r.dtype == torch.float32:
                        wkv6_cuda.backward_fp32(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    elif r.dtype == torch.bfloat16:
                        wkv6_cuda.backward_bf16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    else:
                        wkv6_cuda.backward_fp16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)

                    gu = torch.sum(gu, 0).view(H, C//H)

                    return (None, None, None, None, gr, gk, gv, gw, gu)
        class RWKV_6_with_state:
            @staticmethod
            def apply(B, T, C, H, S, s_map, r, k, v, w, u, s):
                with torch.no_grad():

                    assert s_map is None or s_map.dtype == torch.int64,"s_map 必须为None 或者是长度为B的，int64类型的数组。"
                    assert (s is None and s_map is None) or (s is not None and s_map is not None),"init_state与s_map必须同时为None 或者同时不为None"
                    assert r.dtype == k.dtype == v.dtype == w.dtype == u.dtype and r.dtype in [torch.float16,torch.float32,torch.bfloat16]," r, k, v, w, u 必须为fp16 fp32 bf16中的一种 并且类型相同"
                    if r.dtype in [torch.float32,torch.bfloat16]:
                        o_dtype = r.dtype
                    else:
                        o_dtype = torch.float32
                    assert r.device == k.device == v.device == w.device == u.device,"what kan i say? 请确保r k v w u在同一设备上，快去检查！"

                    y  = torch.empty((B, T, C), device=r.device, dtype=o_dtype, memory_format=torch.contiguous_format)
                    ys = torch.empty((B, H, head_size, head_size), device=r.device, dtype=o_dtype, memory_format=torch.contiguous_format)

                    wkv6_cuda.cuda_forward_with_state_bf16(B, T, C, H, S, s_map, r, k, v, w, u, s)
                
                return y, ys 

        self.head_size = head_size
        self.normal_kernenl = RWKV_6
        self.kernel_with_state = RWKV_6_with_state
    def __call__(self,r, k, v, w, u, return_state=False, init_state=None, state_map=None):
        B,T,C = r.shape
        assert C % self.head_size == 0
        H = C // self.head_size
        if not isinstance(u,torch.Tensor):
            u = u.value

        assert r.is_cuda
        assert k.is_cuda
        assert v.is_cuda
        assert w.is_cuda
        assert u.is_cuda

        if isinstance(r,torch.Tensor):
           
            assert r.device == k.device== v.device == w.device== u.device
        else:
            r.get_device() == k.get_device() == v.get_device() == w.get_device() == u.get_device()

        assert r.dtype == k.dtype == v.dtype == w.dtype  == u.dtype
        
        if init_state is not None:
            assert len(init_state.shape) in [3,4], "init_state 的形状必须为(state_kinds /*<= Batch_size*/,num_heads,head_size,head_size) 或者(num_heads,head_size,head_size)"
            if len(init_state.shape) == 3: init_state = init_state[None, :]
            assert init_state.shape[1:] == (H,self.head_size,self.head_size) and init_state.shape[0] <= B, "init_state 的形状必须为(state_kinds /*<= Batch_size*/,num_heads,head_size,head_size) 或者(num_heads,head_size,head_size)"

            if r.dtype in [ torch.float32,torch.bfloat16]:
                s_dtype = r.dtype
            else:
                s_dtype = torch.float32
            
            assert init_state.dtype == s_dtype,f"init_state的数值类型应为: {s_dtype}"



        if state_map is not None:
            if isinstance(state_map,list):
                state_map = torch.tensor(state_map,dtype=torch.int64)
            elif isinstance(state_map,torch.Tensor):
                assert state_map.dtype in [torch.int32,torch.int64],"state_map是一个长度为Batch_Size的int64类型的映射数组"
                state_map = state_map.to(torch.int64)
            assert state_map.shape == (B,),"state_map的shape必须为(Batch_Size,)"
        
            

        if return_state:
            if init_state is None:
                assert state_map is None,"您必须在指定了init_state的情况下才能使用state_map"
            else:
               

                n_state = init_state.shape[0]
                if state_map is None:
                    assert n_state == 1 or n_state == B,"我无法为您推断state_map的形状，请手动指定。"
                    if n_state == 1:
                        state_map = torch.tensor([0] * n_state,dtype=torch.int64)
                    elif n_state == B:
                        state_map = torch.tensor([i for i in range(B)],dtype=torch.int64)
                    else:
                        assert False,"未实现"
                else:
                    assert state_map.shape == (B,),"state_map的形状必须为(batch_size,)"
                    assert (state_map >= 0).all() and (state_map < n_state).all(),f"state_map的取值范围为[0,{n_state})之间的整数，您的输入显然不满足。"
            
            o, ys = self.kernel_with_state(B, T, C, H, init_state is not None,state_map, r, k, v, w, u, init_state)
            return o, ys
        else:
            o = self.normal_kernenl.apply(B, T, C, H, r, k, v, w, u)
            return o, None
    