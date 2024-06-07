
import os 
assert 'KERAS_BACKEND' in os.environ.keys()

from bert4keras3.backend import ops,keras
from bert4keras3.layers import Layer,Dense,Embedding,LayerNormalization
Model = keras.Model
from rwkv_kernel import RWKVKernelOperator,OPSKernelOperator
from bert4keras3.models import Transformer


class DecomposerDense(Layer):
    def __init__(self,hidden_size,decomposer_size,use_bias=False,name="decomposed_dense"):
        super(DecomposerDense,self).__init__(name=name)
        self.hidden_size = hidden_size
        self.decomposer_size = decomposer_size
        self.use_bias = use_bias
    def build(self, input_shape):
        super().build(input_shape)
        self.dense_a = Dense(self.decomposer_size,activation='tanh',use_bias=False,name="dense_a")
        self.dense_b = Dense(self.hidden_size,use_bias=self.use_bias,name="dense_b")
    def call(self,inputs):
        x = self.dense_a(inputs)
        o = self.dense_b(x)
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
            'hidden_size' : self.hidden_size,
            'decomposer_size' : self.decomposer_size,
            'use_bias' : self.use_bias
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeShift(Layer):
    def __init__(self,name="time_shift"):
        super(TimeShift, self).__init__(name=name)
    def call(self, inputs):
        x = ops.pad(inputs,[[0,0],[1,0],[0,0]],constant_values=0.)[:,:-1,:]
        o = x - inputs
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
def relu_square(x):
    return ops.square(ops.relu(x))

class ChannelMix(Layer):
    def __init__(self,hidden_size,expand_size,name="channel_mix"):
        super(ChannelMix, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.expand_size = expand_size
        self.supports_masking = True
    def call(self, inputs):
        xx = self.timeshift(inputs)
        xk = inputs + xx * ops.reshape(self.time_mix_k,(1,1,self.hidden_size))
        xr = inputs + xx * ops.reshape(self.time_mix_r,(1,1,self.hidden_size))

        k = self.dense_key(xk)
        r = self.dense_receptance(xr)
        kv = self.dense_value(k)
        o = r * kv
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.time_mix_k  = self.add_weight(shape=(self.hidden_size,),name="time_mix_k")
        self.time_mix_r  = self.add_weight(shape=(self.hidden_size,),name="time_mix_r")
        self.timeshift = TimeShift()
        self.dense_key = Dense(self.expand_size,activation=relu_square,use_bias=False,name="dense_k")
        self.dense_value = Dense(self.hidden_size,use_bias=False,name="dense_v")
        self.dense_receptance = Dense(self.hidden_size,activation=ops.sigmoid,use_bias=False,name="dense_r")
        self.hidden_size = self.hidden_size
    def get_config(self):
        config = {
            'hidden_size':self.hidden_size,
            'expand_size':self.expand_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class GroupNorm(Layer):
    def __init__(self,hidden_size,head_size,epsilon=64*1e-5,name="group_norm"):
        super(GroupNorm,self).__init__(name=name)

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.epsilon =epsilon
        assert hidden_size % head_size == 0

    def call(self,inputs):
        B,T,C = inputs.shape
        x = ops.reshape(inputs,(B,T,self.num_heads,self.head_size))
        x =  ops.reshape(self.scale,(1,1,self.num_heads,self.head_size)) * self.group_ln(x) +  ops.reshape(self.offset,(1,1,self.num_heads,self.head_size))
        o = ops.reshape(x,(B,T,C))
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight(shape=(self.num_heads,self.head_size))
        self.offset = self.add_weight(shape=(self.num_heads,self.head_size))
        self.group_ln = LayerNormalization(epsilon=64*1e-5)
    def get_config(self):
        config = {
            'head_size':self.head_size,
            'hidden_size':self.hidden_size,
            'epsilon':self.epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeMix(Layer):
    def __init__(self,layer_idx,rwkv_kernel,hidden_size,decomposer_size,head_size,name="time_mix"):
        super(TimeMix, self).__init__(name=name)
        assert head_size % 4 ==0
        assert head_size % head_size == 0
        num_heads = hidden_size // head_size    
        self.layer_idx = layer_idx
        self.head_size= head_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.supports_masking = True
        self.rwkv_kernel = rwkv_kernel
        self.decomposer_size = decomposer_size
    def call(self, inputs,mask=None):
        if len(inputs)==2:
            inputs,mask = inputs[:]
        else:
            inputs,mask = inputs[0],None
        x_shift = self.timeshift(inputs)
    
        x_mix = inputs + x_shift * ops.reshape(self.time_mix_x,(1, 1, self.hidden_size))


        xr,xk,xv,xw,xg = self.dense_xr(x_mix),self.dense_xk(x_mix),self.dense_xv(x_mix),self.dense_xw(x_mix),self.dense_xg(x_mix)

        xr = inputs + x_shift * xr
        xk = inputs + x_shift * xk
        xv = inputs + x_shift * xv
        xw = inputs + x_shift * xw
        xg = inputs + x_shift * xg
        
        r,k,v,w,g = self.dense_r(xr),self.dense_k(xk),self.dense_v(xv),self.dense_w(xw),self.dense_g(xg)
        if mask!=None:
            mask = ops.cast(mask,inputs.dtype)[...,None]
            xk *= mask
            w = w*mask+(1-mask)*-1e9
            

        x,output_state = self.rwkv_kernel(r,k,v,w,self.time_faaaa)


        x = self.group_norm(x)

        o = self.dense_o(x * g)
        
        return [o]
    def build(self, input_shape):
        super().build(input_shape)
        self.timeshift = TimeShift(name="time_shift")
        self.dense_xr = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xr")
        self.dense_xw = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xw")
        self.dense_xk = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xk")
        self.dense_xv = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xv")
        self.dense_xg = DecomposerDense(self.hidden_size, self.decomposer_size,use_bias=True,name="decomposed_dense_xg")
        
        self.time_mix_x  = self.add_weight(shape=(self.hidden_size,),name="time_mix_x")

        self.dense_r = Dense(self.hidden_size,use_bias=False,name="dense_r")
        self.dense_k = Dense(self.hidden_size,use_bias=False,name="dense_k")
        self.dense_v = Dense(self.hidden_size,use_bias=False,name="dense_v")
        self.dense_w = DecomposerDense(self.hidden_size, self.head_size,use_bias=True,name="decomposed_dense_w")
        self.dense_g = Dense(self.hidden_size,activation=ops.silu,use_bias=False,name="dense_g")
        
        self.time_faaaa = self.add_weight(shape=(self.num_heads,self.head_size),name="time_faaaa")
        
        self.group_norm = GroupNorm(self.hidden_size,self.head_size,name="group_ln")
        self.dense_o = Dense(self.hidden_size,use_bias=False,name="dense_o")
    def get_config(self):
        config = {
           'layer_idx': self.layer_idx , 
           'head_size' : self.head_size,
           'hidden_size':self.hidden_size,
           'num_heads':self.num_heads,
           'rwkv_kernel':self.rwkv_kernel,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RWKV_Block(Layer):
    def __init__(self,layer_idx,rwkv_kernel,hidden_size,decomposer_size,head_size,expand_size,name="rwkv_block"):
        super(RWKV_Block, self).__init__(name=name)
        self.layer_norm_ffn = LayerNormalization(name="ffn_ln",epsilon=1e-5)
        self.layer_norm_att = LayerNormalization(name="att_ln",epsilon=1e-5)
        self.time_mix = TimeMix(layer_idx,rwkv_kernel,hidden_size,decomposer_size, head_size,name=f"time_mix")
        self.channel_mix = ChannelMix(hidden_size, expand_size,name="channel_mix")
        self.layer_idx = layer_idx
    def call(self, inputs):
        x = inputs
        
        x_att = self.layer_norm_att(x)
        o_att = self.time_mix(x_att)
        x = x + o_att

        x_ffn = self.layer_norm_ffn(x)
        o_ffn = self.channel_mix(x_ffn)
        x = x + o_ffn

        return x 
    


class EmbeddingLayer(Layer):
    def __init__(self,vocabulary_size,embedding_size,name="embedding"):
        super(EmbeddingLayer, self).__init__(name=name)
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
    def call(self,inputs):
        x = self.embedding(inputs)
        o = self.layer_norm(x)
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.layer_norm = LayerNormalization(epsilon=1e-5)
        self.embedding = keras.layers.Embedding(self.vocabulary_size,self.embedding_size,mask_zero=True)
    def get_config(self):
        config = {
            'vocabulary_size':self.vocabulary_size,
            'embedding_size':self.embedding_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class OutputLayer(Layer):
    def __init__(self,vocabulary_size,name="output_layer"):
        super(OutputLayer,self).__init__(name=name)
        self.vocabulary_size = vocabulary_size

    def call(self,inputs):
        x = self.layer_norm(inputs)
        o = self.output_layer(x)
        return o
    def compute_output_shape(self, input_shape):
        return list(input_shape[:2])+[self.vocabulary_size]
    def build(self, input_shape):
        super().build(input_shape)
        self.layer_norm = LayerNormalization(name="output_ln",epsilon=1e-5)
        self.output_layer = Dense(self.vocabulary_size,use_bias=False)
    def get_config(self):
        config = {
            'vocabulary_size':self.vocabulary_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class RWKV6(Model):
    def __init__(self,num_layers,vocabulary_size,hidden_size,sequence_length,decomposer_size,head_size,expand_size,name="rwkv6"):
        super(RWKV6, self).__init__(name=name)
        self.embedding_layer = EmbeddingLayer(vocabulary_size, hidden_size)
        self.output_layer = OutputLayer(vocabulary_size,name="output")
        rwkv_kernel = RWKVKernelOperator(head_size, sequence_length)
        self.rwkv_blocks = [RWKV_Block(i,rwkv_kernel,hidden_size, decomposer_size, head_size, expand_size,name=f"block_{i}") for i in range(num_layers)]
    def call(self, inputs):
        x = self.embedding_layer(inputs)

        for idx,block_layer in enumerate(self.rwkv_blocks):
            x = block_layer(x)
        
        o = self.output_layer(x)

        return o


class RWKV_V6(Transformer):
    def __init__(self,decomposer_size,with_lm=True,use_cuda_op=True,**kwargs):
        super().__init__(**kwargs)
        self.decomposer_size = decomposer_size
        self.with_lm = with_lm
        if use_cuda_op:
            self.rwkv_kernal = RWKVKernelOperator(self.attention_head_size, self.sequence_length)
        else:
            self.rwkv_kernal = OPSKernelOperator(self.attention_head_size, self.sequence_length)

    def get_inputs(self):
        x_in = self.apply(
            layer=keras.Input, shape=(self.sequence_length,), name='Input-Token',dtype='int32'
        )
        return  [x_in]
    def get_mask(self):
        if self.attention_bias is None:

            def mask(s):
                return ops.not_equal(s,0)

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=keras.layers.Lambda,
                function=mask,
                name='RKKV-Mask'
            )

        return self.attention_bias
    def apply_embeddings(self, inputs):
        x = inputs.pop(0)
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )

        x = self.apply(
            inputs = x,
            layer = LayerNormalization,
            epsilon=1e-5,
            name='Embedding-LN'
        )
        return x
    def apply_main_layers(self, inputs, index):
        """RWKV的主体是基于Self-Attention的模块
        顺序：LN-->Time-Mix --> Add  --> LN --> Channal-Mix --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        timemix_name = 'RWKV-%d-TimeMix' % index
        channelmix_name = 'RWKV-%d-ChannelMix' % index

        xi = x

        x = self.apply(
            inputs = x,
            layer = LayerNormalization,
            epsilon=1e-5,
            name='%s-Norm' % timemix_name
        )
        mask = self.get_mask()
        x = self.apply(
            inputs = [x,mask],
            layer = TimeMix,
            layer_idx = index,
            rwkv_kernel = self.rwkv_kernal,
            hidden_size = self.hidden_size,
            head_size = self.attention_head_size,
            decomposer_size = self.decomposer_size,
            name = timemix_name
        )[0]

        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % timemix_name
        )

        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % timemix_name
        )

        xi = x

        x = self.apply(
            inputs = x,
            layer = LayerNormalization,
            epsilon=1e-5,
            name='%s-Norm' % channelmix_name
        )
        
        x = self.apply(
            inputs = x,
            layer = ChannelMix,
            hidden_size = self.hidden_size,
            expand_size = self.intermediate_size,
            name = channelmix_name
        )

        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % channelmix_name
        )

        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % channelmix_name
        )
        return x
    def apply_final_layers(self, x):
        x = self.apply(
            inputs = x,
            layer = LayerNormalization,
            epsilon=1e-5,
            name='Out-LN'
        )
        if self.with_lm:
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='RWKV-LM'
                )
        return x
if __name__ == '__main__':
    import numpy as np
    bz, seq_len,hd_sz,head_size = 4,16,128,8
    num_layers = 8
    vocabulary_size = 1140
    decomposer_size = 64
    expand_size = 256
    
    model = RWKV6(num_layers, vocabulary_size, hd_sz, seq_len, decomposer_size, head_size, expand_size)
    inputs = ops.convert_to_tensor(np.random.uniform(size=(bz, seq_len),low=0,high=100).astype('int64'))
    outputs = model(inputs)
