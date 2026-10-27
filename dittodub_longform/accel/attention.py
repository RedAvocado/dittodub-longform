# Native MMA helper and accumulation layout are Copyright Apple Inc.
# Distributed under the MIT license in MLX-LICENSE.
"""Native-order QKV projection, bias and rotary work in two GPU launches."""
import math
from pathlib import Path
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import Attention, RotaryEmbedding
from .rotary import KernelRotary

HEADER = (Path(__file__).parent / 'mlx-steel-mma.h').read_text()
MATMUL = mx.fast.metal_kernel(
    name='ming_qkv_mma_v5', input_names=['x', 'wq', 'wk', 'wv'], output_names=['partial'],
    header=HEADER,
    source='''
    using MMA=mlx::steel::BlockMMA<float,float,8,32,16,1,4,false,true,20,20>;
    threadgroup float As[8*20];
    threadgroup float Bs[32*20];
    uint lane=thread_index_in_simdgroup, group=simdgroup_index_in_threadgroup;
    uint lid=group*32+lane, col=threadgroup_position_in_grid.x*32;
    uint part=threadgroup_position_in_grid.z%2;
    uint projection=threadgroup_position_in_grid.z/2;
    auto weight=projection==0 ? wq : (projection==1 ? wk : wv);
    MMA mma(group,lane);
    for(uint kk=part*512;kk<(part+1)*512;kk+=16){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint ar=lid/16, ac=lid%16, br=lid/4, bc=(lid%4)*4;
        As[ar*20+ac]=ar<M ? x[ar*1024+kk+ac] : 0.0f;
        #pragma clang loop unroll(full)
        for(uint c=0;c<4;c++) Bs[br*20+bc+c]=float(weight[(col+br)*1024+kk+bc+c]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(As,Bs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.store_result_safe(partial+(projection*2+part)*M*1024+col,1024,short2(32,M));
    ''',
)
REDUCE_ROPE = mx.fast.metal_kernel(
    name='ming_qkv_rope_sum_v4', input_names=['partial','bq','bk','bv','cosine','sine'],
    output_names=['q','k','v'],
    header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
    source='''
    uint pair=thread_position_in_grid.x;
    if (pair>=M*512) return;
    uint i=pair*2;
    float qe=0.0f, qo=0.0f, ke=0.0f, ko=0.0f, ve=0.0f, vo=0.0f;
    for (uint p=0;p<2;p++) {
        qe=qe+partial[p*M*1024+i]; qo=qo+partial[p*M*1024+i+1];
        ke=ke+partial[(2+p)*M*1024+i]; ko=ko+partial[(2+p)*M*1024+i+1];
        ve=ve+partial[(4+p)*M*1024+i]; vo=vo+partial[(4+p)*M*1024+i+1];
    }
    uint channel=i%1024, row=i/1024;
    qe=qe+float(bq[channel]); qo=qo+float(bq[channel+1]);
    ke=ke+float(bk[channel]); ko=ko+float(bk[channel+1]);
    v[i]=ve+float(bv[channel]); v[i+1]=vo+float(bv[channel+1]);
    uint phase=(row%SEQ)*32+(channel%64)/2;
    float c=cosine[phase], s=sine[phase];
    float qec=qe*c, qos=qo*s, qes=qe*s, qoc=qo*c;
    float kec=ke*c, kos=ko*s, kes=ke*s, koc=ko*c;
    uint dest=(((row/SEQ)*16+channel/64)*SEQ+row%SEQ)*64+channel%64;
    q[dest]=qec-qos; q[dest+1]=qes+qoc;
    k[dest]=kec-kos; k[dest+1]=kes+koc;
    ''',compile_options={'math_mode':'safe'},
)


def project_rotary(module,x,rope):
    b,s,_=x.shape
    m=b*s
    key=(s,0,id(rope.inv_freq))
    cache=rope.__dict__.setdefault('_constant_cache',{})
    if key not in cache:
        frequencies=mx.outer(mx.arange(s,dtype=mx.float32),rope.inv_freq.astype(mx.float32))
        cosine,sine=mx.cos(frequencies)[None,None,:,:],mx.sin(frequencies)[None,None,:,:]
        mx.eval(cosine,sine)
        if len(cache)>=8:cache.clear()
        cache[key]=(cosine,sine)
    cosine,sine=cache[key]
    partial=MATMUL(inputs=[x,module.to_q.weight,module.to_k.weight,module.to_v.weight],
                   template=[('M',m)],grid=(32*32,2,12),threadgroup=(32,2,2),
                   output_shapes=[(3,2,m,1024)],output_dtypes=[mx.float32])[0]
    q,k,v=REDUCE_ROPE(inputs=[partial,module.to_q.bias,module.to_k.bias,module.to_v.bias,cosine,sine],
                      template=[('M',m),('SEQ',s)],grid=(m*512,1,1),threadgroup=(256,1,1),
                      output_shapes=[(b,16,s,64),(b,16,s,64),x.shape],output_dtypes=[mx.float32]*3)
    return q,k,v.reshape(b,s,16,64).transpose(0,2,1,3)


class FusedRotaryAttention(Attention):
    def __call__(self,x,mask=None,rope=None):
        if (x.dtype!=mx.float32 or x.ndim!=3 or x.shape[-1]!=1024
                or not 1<x.size//1024<=6 or self.heads!=16 or self.dim_head!=64
                or type(rope) not in (RotaryEmbedding,KernelRotary) or rope.inv_freq.shape!=(32,)
                or any(layer.weight.dtype!=mx.bfloat16 or layer.weight.shape!=(1024,1024)
                       or 'bias' not in layer for layer in (self.to_q,self.to_k,self.to_v))):
            return super().__call__(x,mask,rope)
        q,k,v=project_rotary(self,x,rope)
        b,s,_=x.shape
        out=mx.fast.scaled_dot_product_attention(q,k,v,scale=1.0/math.sqrt(64))
        out=out.transpose(0,2,1,3).reshape(b,s,1024)
        out=self.to_out[1](self.to_out[0](out))
        if mask is not None:out=mx.where(mask[...,None],out,0.0)
        return out


def install(model):
    for root in (model.flowloss.cfm.model,model.linear_proj_audio):
        for block in root.blocks:
            block.attn.__class__=FusedRotaryAttention
