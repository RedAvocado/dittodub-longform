# Native MMA helpers are Copyright Apple Inc.; see MLX-LICENSE.
"""Pinned MLX FP32 audio matrix arithmetic with BF16 weights loaded tile by tile."""
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

HEADER = (Path(__file__).parent / 'mlx-steel-mma.h').read_text()
MATMUL = mx.fast.metal_kernel(
    name='ming_audio_mma_v2', input_names=['x', 'weight'], output_names=['partial'],
    header=HEADER,
    source='''
    using MMA = mlx::steel::BlockMMA<float, float, BM, BN, 16, 2, 2,
                                   false, true, 20, 20>;
    threadgroup float As[BM * 20];
    threadgroup float Bs[BN * 20];
    uint lane = thread_index_in_simdgroup;
    uint group = simdgroup_index_in_threadgroup;
    uint lid = group * 32 + lane;
    uint row = threadgroup_position_in_grid.y * BM;
    uint col = threadgroup_position_in_grid.x * BN;
    uint part = threadgroup_position_in_grid.z;
    MMA mma(group, lane);
    for (uint kk = part * KSTEP; kk < (part+1==PARTS ? K : (part+1)*KSTEP); kk += 16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint VA = BM/8, VB = BN/8;
        uint ar=lid/(16/VA), ac=(lid%(16/VA))*VA;
        uint br=lid/(16/VB), bc=(lid%(16/VB))*VB;
        #pragma clang loop unroll(full)
        for (uint c=0;c<VA;c++) As[ar*20+ac+c]=row+ar<M ? x[(row+ar)*K+kk+ac+c] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint c=0;c<VB;c++) Bs[br*20+bc+c]=float(weight[(col+br)*K+kk+bc+c]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(As, Bs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.store_result_safe(partial+part*M*N+row*N+col, N,
                          short2(BN, min(uint(BM), uint(M)-row)));
    ''',
)
REDUCE = mx.fast.metal_kernel(
    name='ming_audio_sum_v2', input_names=['partial', 'bias'], output_names=['out'],
    header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
    source='''
    uint i=thread_position_in_grid.x;
    if(i>=M*N) return;
    float value=PARTS==1 ? partial[i] : 0.0f;
    #pragma clang loop unroll(disable)
    for(uint p=0;p<(PARTS==1 ? 0 : PARTS);p++) value=value+partial[p*M*N+i];
    out[i]=HAS_BIAS ? value+float(bias[i%N]) : value;
    ''',
)


def mixed_linear(x, weight, bias):
    n,k=weight.shape
    m=x.size//k
    split = k>=max(m,n) and k>=128 and ((m+15)//16)*((n+15)//16)<=2048
    if split:
        raw=(k//16)//(((m+31)//32)*((n+31)//32))
        parts=min(max(2,1 << max(0,(raw-1).bit_length())),32)
        bm,bn=16 if m<40 else 32,32
    else:
        bm,bn,parts=64,64,1
    assert k%16==0 and n%bn==0 and m>1
    p=MATMUL(inputs=[x,weight],template=[('M',m),('N',n),('K',k),('BM',bm),('BN',bn),('PARTS',parts),('KSTEP',(k//16//parts)*16)],
             grid=((n//bn)*32,((m+bm-1)//bm)*2,parts*2),threadgroup=(32,2,2),
             output_shapes=[(parts,m,n)],output_dtypes=[mx.float32])[0]
    return REDUCE(inputs=[p,bias if bias is not None else weight],template=[('M',m),('N',n),('PARTS',parts),('HAS_BIAS',bias is not None)],
                  grid=(m*n,1,1),threadgroup=(256,1,1),
                  output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class MixedLinear(nn.Linear):
    def __call__(self,x):
        n,k=self.weight.shape
        if (x.dtype==mx.float32 and self.weight.dtype==mx.bfloat16
                and 1<x.size//k and k%64==0 and n%64==0):
            return mixed_linear(x,self.weight,self.get('bias'))
        return super().__call__(x)


def install(model):
    for root in (model.audio,):
        for _,module in root.named_modules():
            if isinstance(module,nn.Linear):
                module.__class__=MixedLinear
                module.__dict__.pop('_promoted',None)
