# Native MMA helper is Copyright Apple Inc.; see MLX-LICENSE.
"""Preserve the aggregator's one-part reduction before its GELU epilogue."""
import mlx.core as mx
from . import feed_forward as base
from mlx_audio.tts.models.bailingmm.bailingmm import FeedForward

HEADER=base.HEADER+"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
struct GeluBias {
    float apply(float value,float bias) const thread {
        float reduced=0.0f+value;
        float v=reduced+bias;
        float cubic=(v*v)*v;
        float y=v+0.044715f*cubic;
        float z=metal::precise::tanh(0.7978845608028654f*y);
        return (0.5f*v)*(1.0f+z);
    }
};
"""
MATMUL = mx.fast.metal_kernel(
    name='ming_aggregator_gelu_v8', input_names=['x', 'weight', 'bias'], output_names=['partial'],
    header=HEADER, source='''
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
        GeluBias epilogue;
        mma.apply_epilogue(bias+col,0,1,epilogue);
    }
    mma.store_result_safe(partial+part*M*N+col,N,short2(BN,M));
    ''', compile_options={'math_mode': 'safe'})


def expand(x, weight, bias):
    m = x.size // 1024
    return MATMUL(inputs=[x, weight, bias.astype(mx.float32)],
                  template=[('M', m), ('N', 4096), ('K', 1024), ('BM', 8), ('BN', 32),
                            ('BK', 16), ('WM', 1), ('WN', 4), ('PARTS', 1)],
                  grid=(128*32, 2, 2), threadgroup=(32, 2, 2),
                  output_shapes=[(*x.shape[:-1], 4096)], output_dtypes=[mx.float32])[0]


class AggregatorFeedForward(base.FusedFeedForward):
    def __call__(self, x):
        if not self.supported(x):
            return FeedForward.__call__(self, x)
        a, b = self.ff[0][0], self.ff[2]
        return b(expand(x, a.weight, a.bias))

    def with_residual(self, x, residual):
        if not self.supported(x) or residual.shape != x.shape or residual.dtype != mx.float32:
            return residual + FeedForward.__call__(self, x)
        a, b = self.ff[0][0], self.ff[2]
        return base.contract_residual(expand(x, a.weight, a.bias), b.weight, b.bias, residual)


def install(model):
    for block in model.linear_proj_audio.blocks:
        block.mlp.__class__ = AggregatorFeedForward
        block.__dict__.pop('_compiled_v6', None)
