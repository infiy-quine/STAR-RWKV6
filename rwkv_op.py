
import os
import pybind11
import importlib
import sys
import sysconfig
import subprocess
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import pjit
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxlib.hlo_helpers import custom_call


kernel_dir_name = 'rwkv_gpu_kernel'
cuda_lib_dir = "/usr/local/cuda/include"
kernel_name = "gpu_ops"

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


class RWKVKernelOperator:
   
    def __init__(self,head_size,max_sequence_length):
        """
        加载或构建rwkv内核
        """
        rwkv_kernel = RWKVKernelOperator._load_or_build_kernel(head_size, max_sequence_length)
        """
        向mlir注册C++算子入口
        """
        for _name, _value in rwkv_kernel.get_rwkv_registrations().items():
            xla_client.register_custom_call_target(_name, _value, platform="gpu")
        """
        定义前向过程算子
        """
        _rwkv_fwd_p = core.Primitive("rwkv_fwd")
        _rwkv_fwd_p.multiple_results = False
        _rwkv_fwd_p.def_impl(partial(xla.apply_primitive, _rwkv_fwd_p))

        """
        构建jax前端数据类型到C++后端的映射
        """
        def element_type_to_descriptor_type_mapping(element_type):
            _element_type_to_descriptor_type_mapping = {
                ir.BF16Type.get(): rwkv_kernel.ElementType.BF16,
                ir.F16Type.get(): rwkv_kernel.ElementType.F16,
                ir.F32Type.get(): rwkv_kernel.ElementType.F32,
            }
            return _element_type_to_descriptor_type_mapping.get(element_type)

        """
        构建前向过程方法
        """
        def rwkv_fwd(r, k, v, w, u):
            y = _rwkv_fwd_p.bind(r, k, v, w, u)
            ctx = r, k, v, w, u
            return y,ctx
        """
        milr调用上面注册的前向过程算子，milr帮助算子创建输出张量
        """
        def _rwkv_fwd_cuda_lowering(ctx, r, k, v, w, u):
            r_type = ir.RankedTensorType(r.type)
            k_type = ir.RankedTensorType(k.type)
            v_type = ir.RankedTensorType(v.type)
            w_type = ir.RankedTensorType(w.type)
            u_type = ir.RankedTensorType(u.type)
            assert all([r_type.element_type == xx.element_type for xx in [k_type,v_type,w_type,u_type]])
            assert all([r_type.shape == xx.shape for xx in [k_type,v_type,w_type]])
            assert r_type.element_type in [ir.F32Type.get(),ir.BF16Type.get(),ir.F16Type.get()]
            bz,seq_len,hd_sz = r_type.shape

            assert hd_sz % head_size == 0
            assert reduce(lambda x,y: x * y, u_type.shape,1) ==hd_sz,"the elements of u (time first) is not equal to hidden_size"
            input_type = r_type.element_type
            
            if input_type in [ir.F32Type.get(),ir.BF16Type.get()]:
                output_type = input_type
            else:
                output_type = ir.F32Type.get()

            opaque = rwkv_kernel.create_rwkv_descriptor(
                bz, seq_len, hd_sz, hd_sz // head_size,
                element_type_to_descriptor_type_mapping(input_type),
                element_type_to_descriptor_type_mapping(output_type),
            )

            out = custom_call(
                b"wkv_forward",
                result_types=[
                    ir.RankedTensorType.get(r_type.shape, output_type),
                ],
                operands=[r, k, v, w, u],
                backend_config=opaque,
                operand_layouts=default_layouts(r_type.shape,k_type.shape,v_type.shape,w_type.shape,u_type.shape),
                result_layouts=default_layouts(r_type.shape),
            ).results
            return out
        """
        将算子绑定到C++
        """
        mlir.register_lowering(
            _rwkv_fwd_p,
            _rwkv_fwd_cuda_lowering,
            platform="gpu",
        )
        """
        定义抽象过程，告知jax输出张量形状与数据类型
        """
        def _rwkv_fwd_abstract(r, k, v, w, u):
            assert all([r.shape == xx.shape for xx in [k, v, w]])
            assert len(r.shape) == 3
            bz,seq_len,channels = r.shape
            assert channels % head_size == 0
            assert seq_len <= max_sequence_length 
            assert reduce(lambda x,y: x * y, u.shape,1) ==channels,"the elements of u (time first) is not equal to hidden_size"

            r_dtype = dtypes.canonicalize_dtype(r.dtype)
            k_dtype = dtypes.canonicalize_dtype(k.dtype)
            v_dtype = dtypes.canonicalize_dtype(v.dtype)
            w_dtype = dtypes.canonicalize_dtype(w.dtype)
            u_dtype = dtypes.canonicalize_dtype(u.dtype)


            assert all([r_dtype == xx for xx in [k_dtype,v_dtype,w_dtype,u_dtype]])
            assert r_dtype in [jnp.float32,jnp.float16,jnp.bfloat16]
            if r_dtype in [jnp.float32,jnp.bfloat16]:
                output_dtype = r_dtype
            else:
                output_dtype = jnp.float32
            return ShapedArray(r.shape, output_dtype, named_shape=r.named_shape)  # output
        
        _rwkv_fwd_p.def_abstract_eval(_rwkv_fwd_abstract)

        _rwkv_bwd_p = core.Primitive("rwkv_bwd")
        _rwkv_bwd_p.multiple_results = True
        _rwkv_bwd_p.def_impl(partial(xla.apply_primitive, _rwkv_bwd_p))
        
        """
            反向传播过程
        """
        def rwkv_bwd(ctx, gy):
            r, k, v, w, u = ctx
            gr, gk, gv, gw, gu = _rwkv_bwd_p.bind(r, k, v, w, u, gy)
            return gr, gk, gv, gw, gu
        """
            反向传播mlir后端
        """
        def _rwkv_bwd_cuda_lowering(ctx,r, k, v, w, u, gy):
            r_type = ir.RankedTensorType(r.type)
            k_type = ir.RankedTensorType(k.type)
            v_type = ir.RankedTensorType(v.type)
            w_type = ir.RankedTensorType(w.type)
            u_type = ir.RankedTensorType(u.type)
            gy_type =  ir.RankedTensorType(gy.type)

            assert all([r_type.element_type == xx.element_type for xx in [k_type,v_type,w_type,u_type]])
            assert all([r_type.shape == xx.shape for xx in [k_type,v_type,w_type,gy_type]])
            assert r_type.element_type in [ir.F32Type.get(),ir.BF16Type.get(),ir.F16Type.get()]
            bz,seq_len,hd_sz = r_type.shape

            assert hd_sz % head_size == 0
            assert reduce(lambda x,y: x * y, u_type.shape,1) ==hd_sz,"the elements of u (time first) is not equal to hidden_size"
            input_type = r_type.element_type
            
            if input_type in [ir.F32Type.get(),ir.BF16Type.get()]:
                output_type = input_type
            else:
                output_type = ir.F32Type.get()
            
            assert output_type == gy_type.element_type

            opaque = rwkv_kernel.create_rwkv_descriptor(
                bz, seq_len, hd_sz, hd_sz // head_size,
                element_type_to_descriptor_type_mapping(input_type),
                element_type_to_descriptor_type_mapping(output_type),
            )

            out = custom_call(
                b"wkv_backward",
                result_types=[
                    ir.RankedTensorType.get(r_type.shape, output_type),#gr
                    ir.RankedTensorType.get(k_type.shape, output_type),#gk
                    ir.RankedTensorType.get(v_type.shape, output_type),#gw
                    ir.RankedTensorType.get(w_type.shape, output_type),#gw
                    ir.RankedTensorType.get(u_type.shape, output_type),#gu
                ],
                operands=[r, k, v, w, u, gy],
                backend_config=opaque,
                operand_layouts=default_layouts(r_type.shape,k_type.shape,v_type.shape,w_type.shape,u_type.shape,gy_type.shape),
                result_layouts=default_layouts(r_type.shape,k_type.shape,v_type.shape,w_type.shape,u_type.shape),
            ).results
            return out
        """
        注册反向传播算子到mlir
        """
        mlir.register_lowering(
            _rwkv_bwd_p,
            _rwkv_bwd_cuda_lowering,
            platform="gpu",
        )
        """
        反向传播的抽象过程
        """
        def _rwkv_bwd_abstract(r, k, v, w, u, gy):
            assert all([r.shape == xx.shape for xx in [k, v, w]])
            assert len(r.shape) == 3
            bz,seq_len,channels = r.shape
            assert channels % head_size == 0
            assert seq_len <= max_sequence_length 
            assert reduce(lambda x,y: x * y, u.shape,1) ==channels,"the elements of u (time first) is not equal to hidden_size"

            r_dtype = dtypes.canonicalize_dtype(r.dtype)
            k_dtype = dtypes.canonicalize_dtype(k.dtype)
            v_dtype = dtypes.canonicalize_dtype(v.dtype)
            w_dtype = dtypes.canonicalize_dtype(w.dtype)
            u_dtype = dtypes.canonicalize_dtype(u.dtype)
            gy_dtype = dtypes.canonicalize_dtype(gy.dtype)

            assert all([r_dtype == xx for xx in [k_dtype,v_dtype,w_dtype,u_dtype]])
            assert r_dtype in [jnp.float32,jnp.float16,jnp.bfloat16]
            if r_dtype in [jnp.float32,jnp.bfloat16]:
                output_dtype = r_dtype
            else:
                output_dtype = jnp.float32
            assert output_dtype == gy_dtype

            outputs =  [ShapedArray(r.shape, output_dtype, named_shape=r.named_shape),
                        ShapedArray(k.shape, output_dtype, named_shape=k.named_shape),
                        ShapedArray(v.shape, output_dtype, named_shape=v.named_shape),
                        ShapedArray(w.shape, output_dtype, named_shape=w.named_shape),
                        ShapedArray(u.shape, output_dtype, named_shape=u.named_shape),
            ]  # output
            return outputs

        _rwkv_bwd_p.def_abstract_eval(_rwkv_bwd_abstract)
        """
        组合算子
        """
        @jax.custom_vjp
        def rwkv_operator(r, k, v, w, u):
            output, _ = rwkv_fwd(r, k, v, w, u)
            return output
        rwkv_operator.defvjp(rwkv_fwd, rwkv_bwd)
        self.rwkv_operator = rwkv_operator

    def __call__(self, r, k, v, w, u):
        return self.rwkv_operator(r, k, v, w, u) 
    @staticmethod
    def _load_or_build_kernel(head_size,max_sequence_length):
        assert head_size % 4 ==0,f"head size必须是4的倍数，而{head_size}显然不是."
        assert isinstance(head_size, int),"你是在搞笑吗？ head_size肯定得是int类型的啊"
        assert isinstance(max_sequence_length, int),"你是在搞笑吗？ max_sequence_length肯定得是int类型的啊"
        assert head_size >0 and max_sequence_length >0,"难绷，head_size与max_sequence_length肯定得是大于0的正整数啊。"
        assert os.path.exists(cuda_lib_dir) and len( os.listdir(cuda_lib_dir))>0,f"请检查{cuda_lib_dir}文件夹是否存在，这个文件本质是是您的cuda library的超链接。"
        kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),kernel_dir_name))
        builds_dir = os.path.join(kernel_dir,'builds')
        assert os.path.exists(kernel_dir),f"找不到{kernel_dir_name}文件夹，请问您的文件是完整的吗？"
        if not os.path.exists(builds_dir): os.mkdir(builds_dir)
        target_dir_name = f"_N_{head_size}_T_{max_sequence_length}"
        target_dir = os.path.join(builds_dir, target_dir_name)
        if not os.path.exists(target_dir): os.mkdir(target_dir)

        def get_cflags():
            getvar = sysconfig.get_config_var
            flags = ['-I' + sysconfig.get_path('include'),
                    '-I' + sysconfig.get_path('platinclude')]
            
            flags.extend(getvar('CFLAGS').split())
            return ' '.join(flags)
        
        def get_suffix():
            getvar = sysconfig.get_config_var
            return getvar('EXT_SUFFIX')

        build_cmds = []
        
        #first, build cuda kernel
        cu_src = os.path.join(kernel_dir,"rwkv_kernels.cu")
        assert os.path.exists(cu_src)
        cu_dst = os.path.join(target_dir,"rwkv_kernels.cu.o")
        kernel_cmd = f"nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3"+\
            f" --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86]"+\
            f" -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c {cu_src} -o {cu_dst} -D _N_={head_size} -D _T_={max_sequence_length}"
        build_cmds.append(kernel_cmd)

        
        so_dst = os.path.join(target_dir,f"{kernel_name}{get_suffix()}")
        if not os.path.exists(so_dst):
            #second, build C++ code.
            cpp_src = os.path.join(kernel_dir,"gpu_ops.cpp")
            cpp_dst = os.path.join(builds_dir,"gpu_ops.cpp.o")
            if not os.path.exists(cpp_dst):
                cpp_cmd = f"c++ -I{cuda_lib_dir} -I{pybind11.get_include()} {get_cflags()}"+\
                    f" -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects"+\
                    f" -o {cpp_dst} -c {cpp_src}"
                build_cmds.append(cpp_cmd)

            #third assembly C++ and cuda
            assembly_cmd = f"c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared  -o {so_dst} {cpp_dst} {cu_dst}"+\
                f" -L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl"
            build_cmds.append(assembly_cmd)

            #finally strip the so library
            strip_cmd = f"strip {so_dst}"
            build_cmds.append(strip_cmd)
            
            print('-------------------starting build kernel -------------------')
            for cmd in build_cmds:
                print('--------------- execute cmd ---------------')
                print(cmd)
                p = subprocess.Popen(cmd,shell=True)
                p.wait()
            print('-------------------build kernel finished -------------------')

        print('loading cuda kernel')
        rwkv_op = importlib.import_module(f"{kernel_dir_name}.builds.{target_dir_name}.{kernel_name}")
        return rwkv_op
    

if __name__ == '__main__':
    bz,seq_len,hd_sz = 1, 16,8
    r = jnp.zeros(shape=(bz,seq_len,hd_sz),dtype='float16')+2
    k = jnp.zeros(shape=(bz,seq_len,hd_sz),dtype='float16')+2
    v = jnp.zeros(shape=(bz,seq_len,hd_sz),dtype='float16')+2
    w = jnp.zeros(shape=(bz,seq_len,hd_sz),dtype='float16')-2
    u = jnp.zeros(shape=(hd_sz,),dtype='float16')+2
    rwkv_op = RWKVKernelOperator(head_size=hd_sz,max_sequence_length=seq_len)
    out = rwkv_op(r,k,v,w,u)

    print(out.dtype)

    def ref_loss(r, k, v, w, u):
        predictions = rwkv_op(r, k, v, w, u)
        return -jnp.mean(predictions**2)

    ref_out = jax.grad(ref_loss,argnums=(0, 1, 2, 3, 4))(r, k, v, w, u)
    print(ref_out)

