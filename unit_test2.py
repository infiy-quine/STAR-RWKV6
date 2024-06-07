
import os
os.environ['KERAS_BACKEND'] = 'torch'
from keras_torch_transport import TorchToKerasConvertor
from rwkv_v6_demo import RWKV
import torch
import numpy as np
import types
import keras
args = types.SimpleNamespace()
keras.config.set_floatx('bfloat16')
args.n_layer = 24
args.n_embd = 2048

args.vocab_size = 65536
args.ctx_len = 64

########################################################################################################
# CUDA Kernel
########################################################################################################

args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change
print(args)
weight_path = r"/home/niconiconi/rwkv_weights/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"

model_params = torch.load(weight_path, map_location="cpu")

rwkv6_torch = RWKV(args)


rwkv6_torch.bfloat16().to('cuda:1')
rwkv6_torch.load_state_dict(model_params)
print(print(sum(p.numel() for p in rwkv6_torch.parameters())))
del model_params

convertor = TorchToKerasConvertor(weight_path)
model_name = "RWKV6-1.6B"
import torch
try:
    os.makedirs(model_name)
except:
    pass
config={}
config[ "vocab_size"]= args.vocab_size
config[ "num_hidden_layers"]=args.n_layer
config[ "num_attention_heads"]=args.n_embd // args. head_size_a   
config[ "hidden_size"]=args.n_embd
config[ "intermediate_size"]=args.dim_ffn
config[ "attention_head_size"]=args.head_size_a
config[ "decomposer_size"]=convertor.state_dicts['blocks.0.att.time_maa_w2'].shape[1]
import json
with open(model_name+'/config.json', 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
from bert4keras3.models import build_transformer_model
from rwkv6_model import RWKV_V6
rwkv_keras = build_transformer_model(
        config_path=model_name+'/config.json',
        model=RWKV_V6,
        return_keras_model=True,
        sequence_length = args.ctx_len,
        with_lm='linear',
    )


enocder_embeding = rwkv_keras.get_layer('Embedding-Token')
enocder_embeding.embeddings.assign(convertor.get_weight('emb.weight'))
convertor._assign_norm(rwkv_keras.get_layer('Embedding-LN'), 
                       convertor.get_weight('blocks.0.ln0.weight'), 
                       convertor.get_weight('blocks.0.ln0.bias'))

for index in range(args.n_layer):
    timemix_name = 'RWKV-%d-TimeMix' % index
    channelmix_name = 'RWKV-%d-ChannelMix' % index
    timemix_ln =rwkv_keras.get_layer('%s-Norm' % timemix_name)
    timemix = rwkv_keras.get_layer(timemix_name)

    channelmix_ln =rwkv_keras.get_layer('%s-Norm' % channelmix_name)
    channelmix = rwkv_keras.get_layer(channelmix_name)
    block_id = index
    convertor._assign_time_mix(index, timemix)
    convertor._assign_channel_mix(index, channelmix)
    convertor._assign_norm(timemix_ln, 
                           convertor.state_dicts[f'blocks.{block_id}.ln1.weight'], 
                           convertor.state_dicts[f'blocks.{block_id}.ln1.bias'])
    convertor._assign_norm(channelmix_ln,
                            convertor.state_dicts[f'blocks.{block_id}.ln2.weight'], 
                            convertor.state_dicts[f'blocks.{block_id}.ln2.bias'])

convertor._assign_norm(rwkv_keras.get_layer('Out-LN'),
                  convertor.state_dicts['ln_out.weight'],
                  convertor.state_dicts['ln_out.bias'])
convertor._assign_dense(rwkv_keras.get_layer('RWKV-LM'), 
                   convertor.state_dicts['head.weight'])

    




bz = 1
inputs = np.random.uniform(size=(bz,args.ctx_len),low=0,high=4000).astype(np.int64)
inputs = torch.tensor(inputs,dtype=torch.int64)

outputs_torch = rwkv6_torch(inputs.cuda(1))
print(inputs)
outputs_jax = rwkv_keras(inputs)
def calculate_mae(a,b):
    a = a.float().detach().cpu().numpy()
    b = b.float().detach().cpu().numpy()
    return np.abs(a - b).mean()

def calculate_ratio(a,b):
    a = a.float().detach().cpu().numpy()
    b = b.float().detach().cpu().numpy()
    return (np.abs(a - b) / (np.abs(a) + 1e-7)).mean()

print("---------- 总体误差如下: ",calculate_mae(outputs_jax,outputs_torch))
rwkv_keras.save_weights(model_name+'/model.weights.h5')
