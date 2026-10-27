# Native GEMV layout is Copyright Apple Inc.; see MLX-LICENSE.
"""Group independent QKV and gate/up GEMV projections without packed weights."""
import mlx.core as mx
from mlx_audio.lm.models.qwen2 import MLP
from mlx_audio.tts.models.bailingmm.bailingmm import MingQwen2Attention
from mlx_audio.lm.models.base import scaled_dot_product_attention
from mlx_audio.lm.models.activations import swiglu

GROUPED=mx.fast.metal_kernel(
    name='ming_grouped_gemv_v7',input_names=['x','w0','w1','w2','b0','b1','b2'],
    output_names=['o0','o1','o2'],compile_options={'math_mode':'fast'},source='''
    uint lane=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup;
    uint global=threadgroup_position_in_grid.x*BM*4+sg*4;
    uint projection=global<N0?0:(global<N0+N1?1:2);
    uint row=global-(projection==0?0:(projection==1?N0:N0+N1));
    auto weight=projection==0?w0:(projection==1?w1:w2);
    auto bias=projection==0?b0:(projection==1?b1:b2);
    auto out=projection==0?o0:(projection==1?o1:o2);
    float result[4]={0};
    for(uint k=lane*4;k<K;k+=128) {
        float values[4];
        #pragma clang loop unroll(full)
        for(uint c=0;c<4;c++)values[c]=x[k+c];
        #pragma clang loop unroll(full)
        for(uint r=0;r<4;r++) {
            #pragma clang loop unroll(full)
            for(uint c=0;c<4;c++)result[r]+=float(weight[(row+r)*K+k+c])*values[c];
        }
    }
    #pragma clang loop unroll(full)
    for(uint r=0;r<4;r++) {
        #pragma clang loop unroll(full)
        for(ushort delta=16;delta>=1;delta>>=1)result[r]+=simd_shuffle_down(result[r],delta);
    }
    if(lane==0) {
        #pragma clang loop unroll(full)
        for(uint r=0;r<4;r++)out[row+r]=HAS_BIAS?result[r]+float(bias[row+r]):result[r];
    }
    ''')


def grouped(x,layers,bias):
    n=[layer.weight.shape[0] for layer in layers];k=x.shape[-1]
    weights=[layer.weight for layer in layers]
    biases=[layer.bias if bias else layer.weight for layer in layers]
    if len(layers)==2:n.append(0);weights.append(weights[0]);biases.append(biases[0])
    bm=8 if n[0]>=4096 else 4
    return GROUPED(inputs=[x,*weights,*biases],
        template=[('N0',n[0]),('N1',n[1]),('K',k),('BM',bm),('HAS_BIAS',bias)],
        grid=(sum(n)//(bm*4)*32,1,bm),threadgroup=(32,1,bm),
        output_shapes=[(*x.shape[:-1],v) for v in n[:2]]+[(*x.shape[:-1],max(1,n[2]))],
        output_dtypes=[mx.float32]*3)[:len(layers)]


class GroupedAttention(MingQwen2Attention):
    def __call__(self,x,mask=None,cache=None):
        layers=(self.q_proj,self.k_proj,self.v_proj)
        if (self.sliding_window is not None or x.shape!=(1,1,896) or x.dtype!=mx.float32 or self.n_heads!=14 or self.n_kv_heads!=2
                or any(layer.weight.dtype!=mx.bfloat16 or layer.weight.shape!=(n,896)
                       or 'bias' not in layer or layer.bias.shape!=(n,)
                       for layer,n in zip(layers,(896,128,128)))):
            return super().__call__(x,mask,cache)
        queries,keys,values=grouped(x,layers,True)
        queries=queries.reshape(1,1,14,64).transpose(0,2,1,3)
        keys=keys.reshape(1,1,2,64).transpose(0,2,1,3)
        values=values.reshape(1,1,2,64).transpose(0,2,1,3)
        if cache is not None:
            queries=self.rope(queries,offset=cache.offset)
            keys=self.rope(keys,offset=cache.offset)
            keys,values=cache.update_and_fetch(keys,values)
        else:
            queries=self.rope(queries);keys=self.rope(keys)
        output=scaled_dot_product_attention(queries,keys,values,cache=cache,scale=self.scale,mask=mask)
        return self.o_proj(output.transpose(0,2,1,3).reshape(1,1,896))


class GroupedMLP(MLP):
    def __call__(self,x):
        layers=(self.gate_proj,self.up_proj)
        if (x.shape!=(1,1,896) or x.dtype!=mx.float32
                or any(layer.weight.dtype!=mx.bfloat16 or layer.weight.shape!=(4864,896)
                       or 'bias' in layer for layer in layers)):
            return super().__call__(x)
        gate,up=grouped(x,layers,False)
        return self.down_proj(swiglu(gate,up))


def install(model):
    for block in model.model.model.layers:
        assert type(block.self_attn) is MingQwen2Attention and type(block.mlp) is MLP
        block.self_attn.__class__=GroupedAttention
        block.mlp.__class__=GroupedMLP
