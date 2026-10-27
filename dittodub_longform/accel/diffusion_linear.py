# Native MMA helper is Copyright Apple Inc.; see MLX-LICENSE.
"""Small-row diffusion matrices without persistent FP32 weight copies.

Keep the native FP32 8x8 MMA order and split-K partitions. Eight-row tiles
avoid arithmetic on unused rows; BF16 weights expand only in threadgroup
memory. Non-split output adds its bias before the single store.
"""
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

HEADER=(Path(__file__).parent/'mlx-steel-mma.h').read_text()
MATMUL=mx.fast.metal_kernel(
    name='ming_ffn_mma_v5',input_names=['x','weight','bias'],output_names=['partial'],
    header=HEADER,
    source='''
    using MMA=mlx::steel::BlockMMA<float,float,BM,BN,BK,WM,WN,
                                 false,true,BK+4,BK+4>;
    threadgroup float As[BM*(BK+4)];
    threadgroup float Bs[BN*(BK+4)];
    uint lane=thread_index_in_simdgroup, group=simdgroup_index_in_threadgroup;
    uint lid=group*32+lane, col=threadgroup_position_in_grid.x*BN;
    uint part=threadgroup_position_in_grid.z;
    MMA mma(group,lane);
    for(uint kk=part*(K/PARTS);kk<(part+1)*(K/PARTS);kk+=BK){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint VA=BM*BK/128<4 ? BM*BK/128 : 4, VB=BN*BK/128<4 ? BN*BK/128 : 4;
        for(uint v=lid*VA;v<BM*BK;v+=128*VA){
            uint r=v/BK, c=v%BK;
            #pragma clang loop unroll(full)
            for(uint j=0;j<VA;j++) As[r*(BK+4)+c+j]=r<M ? x[r*K+kk+c+j] : 0.0f;
        }
        for(uint v=lid*VB;v<BN*BK;v+=128*VB){
            uint r=v/BK, c=v%BK;
            #pragma clang loop unroll(full)
            for(uint j=0;j<VB;j++) Bs[r*(BK+4)+c+j]=float(weight[(col+r)*K+kk+c+j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(As,Bs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(PARTS==1) {
        mlx::steel::TransformAdd<float,float> epilogue(1.0f,1.0f);
        mma.apply_epilogue(bias+col,0,1,epilogue);
    }
    mma.store_result_safe(partial+part*M*N+col,N,short2(BN,M));
    ''')

REDUCE=mx.fast.metal_kernel(
    name='ming_ffn_sum_v5',input_names=['partial','bias'],output_names=['out'],
    header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
    source='''
    uint i=thread_position_in_grid.x;
    if(i>=M*N)return;
    float value=0.0f;
    #pragma clang loop unroll(disable)
    for(uint p=0;p<PARTS;p++)value=value+partial[p*M*N+i];
    out[i]=value+bias[i%N];
    ''',compile_options={'math_mode':'safe'})


def linear(x,weight,bias,tile=(8,32,16,1,4)):
    n,k=weight.shape;m=x.size//k
    parts=8 if (n,k)==(1024,4096) else 2 if (n,k)==(1024,1024) else 1
    bm,bn,bk,wm,wn=tile
    partial=MATMUL(inputs=[x,weight,bias.astype(mx.float32)],
        template=[('M',m),('N',n),('K',k),('BM',bm),('BN',bn),('BK',bk),('WM',wm),('WN',wn),('PARTS',parts)],
        grid=(n//bn*32,2,parts*2),threadgroup=(32,2,2),
        output_shapes=[(parts,m,n)],output_dtypes=[mx.float32])[0]
    if parts==1:return partial.reshape(*x.shape[:-1],n)
    return REDUCE(inputs=[partial,bias.astype(mx.float32)],template=[('M',m),('N',n),('PARTS',parts)],
        grid=(m*n,1,1),threadgroup=(256,1,1),output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class DiffusionLinear(nn.Linear):
    def __call__(self,x):
        if (x.dtype==mx.float32 and self.weight.dtype==mx.bfloat16 and 1<x.size//x.shape[-1]<=6
                and self.weight.shape in ((4096,1024),(1024,4096),(1024,1024))
                and 'bias' in self and self.bias.shape==(self.weight.shape[0],)):
            return linear(x,self.weight,self.bias)
        return super().__call__(x)


def install(model):
    for block in model.flowloss.cfm.model.blocks:
        for layer in (block.mlp.ff[0][0],block.mlp.ff[2],block.attn.to_out[0]):
            layer.__class__=DiffusionLinear
            layer.__dict__.pop('_promoted',None)
