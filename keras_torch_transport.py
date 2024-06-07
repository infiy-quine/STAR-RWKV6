
import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import keras
import re
from keras import saving
from rwkv6_model import RWKV6


class TorchToKerasConvertor:
    def __init__(self,pth_path):
        assert os.path.exists(pth_path)
        self.pth_path = pth_path

        self.state_dicts = torch.load(self.pth_path,map_location='cpu')
        self.vocabulary_size,self.embedding_size = self.state_dicts['emb.weight'].shape
        self.num_layers = self._get_num_layers(self.state_dicts.keys())
        self.num_heads,self.head_size = self.state_dicts['blocks.0.att.time_faaaa'].shape
        
        self.hidden_size = self.state_dicts['blocks.0.att.time_decay'].shape[-1]
        assert self.hidden_size == self.embedding_size == self.num_heads * self.head_size
        
        self.decomposer_size = self.state_dicts['blocks.0.att.time_maa_w2'].shape[1]
        self.expand_size = self.state_dicts['blocks.0.ffn.key.weight'].shape[0]
        print('开始转换模型，模型参数如下: ')
        print(f'num_layers={self.num_layers}, num_heads={self.num_heads}, head_size={self.head_size}')
        print(f'expand_size={self.expand_size}, decomposer_size={self.decomposer_size}')
        print('--------------------------')
        for key,value in self.state_dicts.items():
            print(f'name: {key}, shape= {value.shape}')
    def _get_num_layers(self,keys):
        num_layers = 0
        for key in keys:
            g = re.match(r"^blocks\.(?P<layer_id>\d+)\.(.+)$", key)
            if g:
                num_layers = max(num_layers,int(g.group('layer_id')) + 1)
        
        return num_layers

    def get_weight(self,name=None,value=None):
        if name is not None:
            value =  self.state_dicts[name]
        value = value.detach().cpu().float().numpy()
        return value
    
    def _assign_embedding(self,model):
        emb_layer = model.embedding_layer
        emb_layer.embedding.embeddings.assign(self.get_weight('emb.weight'))
        self._assign_norm(emb_layer.layer_norm, self.get_weight('blocks.0.ln0.weight'), self.get_weight('blocks.0.ln0.bias'))
    
    def _assign_decomposer_dense(self,layer,w1,w2,bias=None):
        layer.dense_a.kernel.assign(w1)
        layer.dense_b.kernel.assign(w2)
        if bias is not None:
            assert layer.dense_b.use_bias
            layer.dense_b.bias.assign(bias)
        else:
            assert not layer.dense_b.use_bias
    def _assign_dense(self,layer,weight,bias=None):
        weight = weight.T
        layer.kernel.assign(weight)

        if bias is not None:
            assert layer.dense_b.use_bias
            layer.bias.assign(bias)
        else:
            assert not layer.use_bias
            
    def _assign_norm(self,layer,weight,bias):
        assert layer.gamma.shape == weight.shape,(layer.name,layer.gamma.shape, weight.shape)
        assert layer.beta.shape == bias.shape,layer.name
        layer.gamma.assign(weight)
        layer.beta.assign(bias)


    def _assign_group_norm(self,layer,weight,bias):
        weight = weight.reshape((self.num_heads,self.head_size))
        bias = bias.reshape((self.num_heads,self.head_size))
        layer.scale.assign(weight)
        layer.offset.assign(bias)

    def _assign_time_mix(self,block_id,time_mix):
        tmp = self.state_dicts[f'blocks.{block_id}.att.time_maa_w1']
        assert tmp.shape[1] % 5 ==0
        ww1,wk1,wv1,wr1,wg1 = torch.reshape(tmp,(tmp.shape[0],5,tmp.shape[1]//5)).unbind(dim=1)
        tmp = self.state_dicts[f'blocks.{block_id}.att.time_maa_w2']
        ww2,wk2,wv2,wr2,wg2 = tmp.unbind(dim=0)

        br = self.state_dicts[f'blocks.{block_id}.att.time_maa_r']
        self._assign_decomposer_dense(time_mix.dense_xr, wr1, wr2,torch.reshape(br,(-1,)))
        bw = self.state_dicts[f'blocks.{block_id}.att.time_maa_w']
        self._assign_decomposer_dense(time_mix.dense_xw, ww1, ww2,torch.reshape(bw,(-1,)))
        bk = self.state_dicts[f'blocks.{block_id}.att.time_maa_k']
        self._assign_decomposer_dense(time_mix.dense_xk, wk1, wk2,torch.reshape(bk,(-1,)))
        bv = self.state_dicts[f'blocks.{block_id}.att.time_maa_v']
        self._assign_decomposer_dense(time_mix.dense_xv, wv1, wv2,torch.reshape(bv,(-1,)))
        bg = self.state_dicts[f'blocks.{block_id}.att.time_maa_g']
        self._assign_decomposer_dense(time_mix.dense_xg, wg1, wg2,torch.reshape(bg,(-1,)))
        
        tmp = self.state_dicts[f'blocks.{block_id}.att.time_maa_x']
        time_mix.time_mix_x.assign(torch.reshape(tmp,(-1,)))

        tmp = self.state_dicts[f'blocks.{block_id}.att.receptance.weight']
        self._assign_dense(time_mix.dense_r, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.att.key.weight']
        self._assign_dense(time_mix.dense_k, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.att.value.weight']
        self._assign_dense(time_mix.dense_v, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.att.gate.weight']
        self._assign_dense(time_mix.dense_g, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.att.output.weight']
        self._assign_dense(time_mix.dense_o, tmp)

        wc1,wc2 = self.state_dicts[f'blocks.{block_id}.att.time_decay_w1'],self.state_dicts[f'blocks.{block_id}.att.time_decay_w2']
        wb = torch.reshape(self.state_dicts[f'blocks.{block_id}.att.time_decay'],(-1,))
        self._assign_decomposer_dense(time_mix.dense_w, wc1, wc2,bias=wb)

        tmp = self.state_dicts[f'blocks.{block_id}.att.time_faaaa']
        time_mix.time_faaaa.assign(tmp)

        ln_w,ln_x = self.state_dicts[f'blocks.{block_id}.att.ln_x.weight'],self.state_dicts[f'blocks.{block_id}.att.ln_x.bias']
        self._assign_group_norm(time_mix.group_norm,ln_w,ln_x)

    def _assign_channel_mix(self, block_id, channel_mix):
        channel_mix.time_mix_k.assign(torch.reshape(self.state_dicts[f'blocks.{block_id}.ffn.time_maa_k'],(-1,)))
        channel_mix.time_mix_r.assign(torch.reshape(self.state_dicts[f'blocks.{block_id}.ffn.time_maa_r'],(-1,)))

        tmp = self.state_dicts[f'blocks.{block_id}.ffn.key.weight']
        self._assign_dense(channel_mix.dense_key, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.ffn.receptance.weight']
        self._assign_dense(channel_mix.dense_receptance, tmp)
        tmp = self.state_dicts[f'blocks.{block_id}.ffn.value.weight']
        self._assign_dense(channel_mix.dense_value, tmp)

    
    def _assign_block(self,block_id,block):
        self._assign_time_mix(block_id, block.time_mix)
        self._assign_channel_mix(block_id, block.channel_mix)
        self._assign_norm(block.layer_norm_att, self.state_dicts[f'blocks.{block_id}.ln1.weight'], self.state_dicts[f'blocks.{block_id}.ln1.bias'])
        self._assign_norm(block.layer_norm_ffn, self.state_dicts[f'blocks.{block_id}.ln2.weight'], self.state_dicts[f'blocks.{block_id}.ln2.bias'])

    def _assign_output(self,model):
        out_layer = model.output_layer
        self._assign_norm(out_layer.layer_norm,self.state_dicts['ln_out.weight'],self.state_dicts['ln_out.bias'])
        self._assign_dense(out_layer.output_layer, self.state_dicts['head.weight'])

    def build_rwkv_from_torch(self):
        #assert not os.path.exists(output_path)
        #assert output_path.endswith(".keras")
        seq_len = 512
        rwkv_model = RWKV6(self.num_layers,self.vocabulary_size,self.hidden_size,seq_len,self.decomposer_size,self.head_size,self.expand_size)
        inputs = torch.zeros(size=(1,seq_len),dtype=torch.int64)
        rwkv_model(inputs)
        self._assign_embedding(rwkv_model)
        for idx,block in enumerate(rwkv_model.rwkv_blocks):
            self._assign_block(idx,block)
        self._assign_output(rwkv_model)
        
        return rwkv_model

    def convert2kears(self,output_path):
        rwkv_model = self.build_rwkv_from_torch()
        if output_path.endswith(".keras"):
            rwkv_model.save(output_path)
        elif output_path.endswith("weights.h5"):
            #pass
            rwkv_model.save_weights(output_path)

        rwkv_model.load_weights(output_path)

    

# convertor = TorchToKerasConvertor("./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")

# convertor.convert2kears("rwkv6_1B6.weights.h5")