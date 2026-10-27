# Native reduction layout is Copyright Apple Inc.; see MLX-LICENSE.
"""Exact attention residual/RMS fusion and bounded diffusion graph reuse.

Only the fixed six-row block is compiled. Its arithmetic lives in separately
qualified custom kernels and unchanged native attention; the solver, random
state and unsupported block configurations remain outside compilation.
"""
import math
import mlx.core as mx
from .attention import project_rotary
from .diffusion_linear import MATMUL
from .feed_forward import FusedDiffusionBlock
from .rotary import KernelRotary
from mlx_audio.tts.models.bailingmm.bailingmm import RotaryEmbedding

REDUCE_RESIDUAL_NORM = mx.fast.metal_kernel(
    name='ming_reduce_residual_norm_v6',
    input_names=['partial','bias','residual','weight','exponent'],
    output_names=['out','normalized'],
    header='#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n',
    compile_options={'math_mode':'safe'}, source='''
    uint lid=thread_position_in_threadgroup.x;
    uint row=threadgroup_position_in_grid.y;
    uint lane=thread_index_in_simdgroup;
    uint group=simdgroup_index_in_threadgroup;
    threadgroup float sums[4];
    threadgroup float inverse;
    float values[8];
    float total=0.0f;
    for(uint block=0;block<2;block++) {
        for(uint j=0;j<4;j++) {
            uint c=block*512+lid*4+j, i=row*1024+c;
            float value=0.0f;
            #pragma clang loop unroll(disable)
            for(uint p=0;p<PARTS;p++) value=value+partial[p*M*1024+i];
            value=value+float(bias[c]);
            value=residual[i]+value;
            values[block*4+j]=value;
            out[i]=value;
            float squared=metal::precise::pow(value,exponent);
            total=squared+total;
        }
    }
    total=0.0f+total;
    total=simd_sum(total);
    if(lane==0)sums[group]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value=lid<4?sums[lid]:0.0f;
    value=simd_sum(value);
    if(lid==0)inverse=metal::precise::rsqrt(value*(1.0f/1024.0f)+1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<2;block++) {
        for(uint j=0;j<4;j++) {
            uint c=block*512+lid*4+j;
            float norm=values[block*4+j]*inverse;
            normalized[row*1024+c]=norm*float(weight[c]);
        }
    }
    ''')


def residual_norm(partial,bias,residual,weight,parts):
    m=residual.size//1024
    return REDUCE_RESIDUAL_NORM(
        inputs=[partial,bias,residual,weight,mx.array(2.0,dtype=mx.float32)],
        template=[('M',m),('PARTS',parts)],grid=(128,m,1),threadgroup=(128,1,1),
        output_shapes=[residual.shape,residual.shape],output_dtypes=[mx.float32,mx.float32])


def dot_partial(x,layer,parts):
    n,k=layer.weight.shape;m=x.size//k
    return MATMUL(inputs=[x,layer.weight,layer.bias.astype(mx.float32)],
        template=[('M',m),('N',n),('K',k),('BM',8),('BN',32),('BK',16),('WM',1),('WN',4),('PARTS',parts)],
        grid=(n//32*32,2,parts*2),threadgroup=(32,2,2),
        output_shapes=[(parts,m,n)],output_dtypes=[mx.float32])[0]


class ResidualNormBlock(FusedDiffusionBlock):
    def supported(self,x,mask,rope):
        if (x.shape!=(2,3,1024) or x.dtype!=mx.float32 or mask is not None
                or type(rope) not in (RotaryEmbedding,KernelRotary) or rope.inv_freq.shape!=(32,)
                or self.attn.heads!=16 or self.attn.dim_head!=64
                or self.norm1.eps!=1e-6 or self.norm2.eps!=1e-6
                or self.norm1.weight.shape!=(1024,) or self.norm2.weight.shape!=(1024,)
                or self.norm2.weight.dtype not in (mx.float32,mx.bfloat16)
                or not self.mlp.supported(x)):
            return False
        return all(layer.weight.dtype==mx.bfloat16 and layer.weight.shape==(1024,1024)
                   and 'bias' in layer and layer.bias.shape==(1024,)
                   for layer in (self.attn.to_q,self.attn.to_k,self.attn.to_v,self.attn.to_out[0]))

    def __call__(self,x,mask,rope):
        # Specialize only the qualified dense diffusion path.
        if not self.supported(x,mask,rope):
            return super().__call__(x,mask,rope)
        q,k,v=project_rotary(self.attn,self.norm1(x),rope)
        y=mx.fast.scaled_dot_product_attention(q,k,v,scale=1.0/math.sqrt(64))
        y=y.transpose(0,2,1,3).reshape(x.shape)
        layer=self.attn.to_out[0]
        p=dot_partial(y,layer,2)
        residual,normalized=residual_norm(p,layer.bias,x,self.norm2.weight,2)
        return self.mlp.with_residual(normalized,residual)


class CompiledResidualNormBlock(ResidualNormBlock):
    def __call__(self,x,mask,rope):
        if not self.supported(x,mask,rope):
            return super().__call__(x,mask,rope)
        key=(id(rope),id(rope.inv_freq),self.norm1.eps,self.norm2.eps,self.attn.heads,self.attn.dim_head)
        cached=self.__dict__.get('_compiled_v6')
        if cached is None or cached[0]!=key:
            # Initialize the small RoPE constants before entering a trace.
            ResidualNormBlock.__call__(self,x,None,rope)
            compiled=mx.compile(lambda value: ResidualNormBlock.__call__(self,value,None,rope),inputs=self)
            cached=(key,compiled)
            self.__dict__['_compiled_v6']=cached
        return cached[1](x)


def install(model,compiled=False):
    for block in model.flowloss.cfm.model.blocks:
        block.__class__=CompiledResidualNormBlock if compiled else ResidualNormBlock
