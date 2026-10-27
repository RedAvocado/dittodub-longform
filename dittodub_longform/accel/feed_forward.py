# Native MMA helper is Copyright Apple Inc.; see MLX-LICENSE.
"""Native MMA with an exactly rounded bias/GELU expansion epilogue.

MLX's compiled gelu_approx lowers its cube to two multiplies. Preserve that
order, precise tanh, and every rounding boundary; neither pow nor FMA is an
equivalent substitution here. The contraction keeps all eight native K sums
before adding bias and the residual, in that order.
"""
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import FeedForward,DiTBlock
from .diffusion_linear import HEADER

GELU_HEADER=HEADER+"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
struct GeluBias {
    float apply(float value,float bias) const thread {
        float v=value+bias;
        float cubic=(v*v)*v;
        float y=v+0.044715f*cubic;
        float z=metal::precise::tanh(0.7978845608028654f*y);
        return (0.5f*v)*(1.0f+z);
    }
};
"""
MATMUL=mx.fast.metal_kernel(name='ming_gelu_mma_v5',input_names=['x','weight','bias'],output_names=['partial'],
    header=GELU_HEADER,compile_options={'math_mode':'safe'},source='''
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
    ''')

REDUCE_RESIDUAL=mx.fast.metal_kernel(
    name='ming_ffn_residual_v5',input_names=['partial','bias','residual'],output_names=['out'],
    header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
    compile_options={'math_mode':'safe'},source="""
    uint i=thread_position_in_grid.x;
    if(i>=M*1024)return;
    float value=0.0f;
    #pragma clang loop unroll(disable)
    for(uint p=0;p<8;p++)value=value+partial[p*M*1024+i];
    value=value+bias[i%1024];
    out[i]=residual[i]+value;
    """)


def expand(x,weight,bias):
    m=x.size//1024
    return MATMUL(inputs=[x,weight,bias.astype(mx.float32)],
        template=[('M',m),('N',4096),('K',1024),('BM',8),('BN',32),('BK',16),('WM',1),('WN',4),('PARTS',1)],
        grid=(128*32,2,2),threadgroup=(32,2,2),
        output_shapes=[(*x.shape[:-1],4096)],output_dtypes=[mx.float32])[0]


def contract_residual(x,weight,bias,residual):
    from .diffusion_linear import MATMUL as DOT
    m=x.size//4096
    p=DOT(inputs=[x,weight,bias.astype(mx.float32)],
        template=[('M',m),('N',1024),('K',4096),('BM',8),('BN',32),('BK',16),('WM',1),('WN',4),('PARTS',8)],
        grid=(32*32,2,16),threadgroup=(32,2,2),
        output_shapes=[(8,m,1024)],output_dtypes=[mx.float32])[0]
    return REDUCE_RESIDUAL(inputs=[p,bias.astype(mx.float32),residual],template=[('M',m)],
        grid=(m*1024,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[mx.float32])[0]


class FusedFeedForward(FeedForward):
    def supported(self,x):
        a,b=self.ff[0][0],self.ff[2]
        return (x.dtype==mx.float32 and x.shape[-1]==1024 and 1<x.size//1024<=6
            and a.weight.dtype==b.weight.dtype==mx.bfloat16
            and a.weight.shape==(4096,1024) and b.weight.shape==(1024,4096)
            and 'bias' in a and 'bias' in b
            and a.bias.shape==(4096,) and b.bias.shape==(1024,))

    def __call__(self,x):
        if not self.supported(x):return super().__call__(x)
        a,b=self.ff[0][0],self.ff[2]
        y=expand(x,a.weight,a.bias)
        return b(y)

    def with_residual(self,x,residual):
        if (not self.supported(x) or residual.dtype!=mx.float32 or residual.shape!=x.shape):
            return residual+super().__call__(x)
        a,b=self.ff[0][0],self.ff[2]
        return contract_residual(expand(x,a.weight,a.bias),b.weight,b.bias,residual)


class FusedDiffusionBlock(DiTBlock):
    def __call__(self,x,mask,rope):
        x=x+self.attn(self.norm1(x),mask=mask,rope=rope)
        return self.mlp.with_residual(self.norm2(x),x)


def install(model):
    for block in model.flowloss.cfm.model.blocks:
        block.mlp.__class__=FusedFeedForward
        block.__class__=FusedDiffusionBlock
