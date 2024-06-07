from bert4keras3.backend import ops,keras,int_shape
class RWKVKernelOperator:
    def __init__(self,head_size,max_sequence_length):

        self.head_size = head_size
        self.max_sequence_length = max_sequence_length
    def __call_parallel__(self,r, k, v, w, u):
        B,L,D = int_shape(w)
        H,S = D//self.head_size,self.head_size
        w = ops.exp(-ops.exp(ops.reshape(w,[B, L, H, S, 1])))
        # 计算注意力机制的组件
        r = ops.reshape(r,[B, L, H, 1, S])
        k = ops.reshape(k,[B, L, H, S, 1])
        v = ops.reshape(v, [B,L, H, 1, S])
        states = [ops.zeros([B, 1, H, S,S])]
        a = k @ v # a: [batch_size, L, H, S, S]
        ua = ops.reshape(u,[1,1,H,S,1]) * a
        for l in range(L-1):
            s = a[:, l][:,None] + w[:, l][:,None] * states[-1]
            
            states += [s]
        #s = a[:, -1] + w[:, -1] * states[-1]
        states = ops.concatenate(states,1)
        
        o= r @ (ua + states)
        return ops.reshape(o,[B,L,-1])

    def __call__(self,r, k, v, w, u, return_state=False, init_state=None, state_map=None):
        B,T,C = int_shape(r)
        assert C % self.head_size == 0
        H = C // self.head_size

        w = ops.reshape(w, [B,T,H,self.head_size,1])
        k = ops.reshape(k, [B,T,H,self.head_size,1])

        v = ops.reshape(v, [B,T,H,1,self.head_size])
        r = ops.reshape(r,[B,T,H,1,self.head_size])
        u = ops.reshape(u, [1,H,self.head_size,1])

        s = ops.zeros((B,H,self.head_size,self.head_size),dtype=u.dtype)

        w = ops.exp(-ops.exp(w))


        y_list = []
        for i in range(T):
            kv_t = k[:,i,:,:,:] @ v[:,i,:,:,:]
            w_t =  w[:,i,:,:,:]
            r_t = r[:,i,:,:,:]
            y_t = r_t @ (u * kv_t + s)

            y_t = ops.reshape(y_t,(B,C))
            s = kv_t +w_t * s
            y_list.append(y_t)
        x = ops.stack(y_list,axis=1)

        x = ops.reshape(x,(B,T,C))
        return x,s