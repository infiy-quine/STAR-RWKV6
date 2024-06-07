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

    def __call__(self,r, k, v, w, u, with_state=False, init_state=None, state_map=None):
        B,T,C = int_shape(r)
        assert C % self.head_size == 0
        H = C // self.head_size

        w = ops.reshape(w, [B,T,H,self.head_size,1])
        k = ops.reshape(k, [B,T,H,self.head_size,1])

        v = ops.reshape(v, [B,T,H,1,self.head_size])
        r = ops.reshape(r,[B,T,H,1,self.head_size])
        u = ops.reshape(u, [1,H,self.head_size,1])


        if init_state is not  None:
            assert len(init_state.shape) in [3,4], "init_state的形状必须为(state_kinds,num_heads,head_size,head_size)"
            if len(init_state.shape) == 3:
                assert init_state.shape == (H,self.head_size,self.head_size),"state_kinds的形状必须为(BatchSize,num_heads,head_size,head_size)"
                init_state = init_state[None,:]
            else:
                assert init_state.shape[1:]  == (H,self.head_size,self.head_size),"state_kinds的形状必须为(BatchSize,num_heads,head_size,head_size)"
                state_kinds = init_state.shape[0]
                assert state_kinds <= B

            if state_map is None:
                state_kinds = init_state.shape[0]
                if state_kinds == 1:
                    state_map = ops.zeros(shape=(B,),dtype=ops.int64)
                elif state_kinds == B:
                    state_map = ops.convert_to_tensor([i for i in range(B)],dtype=ops.int64)
                else:
                    raise ValueError("无法为您推断state_map的形状，请您手动指定state_map")

            else:
                if isinstance(state_map,list):
                    state_map = ops.convert_to_tensor(state_map,dtype=ops.int64)
                assert state_map.dtype in [ops.int32,ops.int64], "state_map为一个int64类型的数组"
                state_map = ops.cast(state_map,ops.int64)

                assert (state_map >=0).all() and (state_map < state_kinds).all(),f"请确保state_map的值域为[0, {state_kinds})"

            s = ops.take(init_state,state_map,axis=0)

        else:
            assert state_map is None
            s = ops.zeros((1,H,self.head_size,self.head_size),dtype=u.dtype)



    


        
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
        if with_state:
            return x,s
        return x,s