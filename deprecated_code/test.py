from torch.utils.cpp_extension import load
kernel_dir_name = 'torch_kernel'

head_size = 64
max_sequence_length = 512
wkv6_cuda = load(name="wkv6", sources=[f"{kernel_dir_name}/wkv6_op.cpp", f"{kernel_dir_name}/wkv6_cuda.cu"],
    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size}", f"-D_T_={max_sequence_length}"])

print(dir(wkv6_cuda))
