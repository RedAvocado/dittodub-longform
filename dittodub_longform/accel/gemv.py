# Accumulation layout adapted from MLX 0.32.4 gemv.h, Copyright Apple Inc.
# Distributed under the MIT license in MLX-LICENSE.
"""Native MLX FP32 GEMV accumulation with BF16 matrix loads converted in registers."""
import mlx.core as mx
import mlx.nn as nn

KERNEL=mx.fast.metal_kernel(
    name='ming_mixed_gemv_v2',input_names=['x','weight','bias'],output_names=['out'],
    source='''
    uint lane=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup;
    uint row=threadgroup_position_in_grid.x*BM*4+sg*4;
    float result[4]={0};
    for(uint k=lane*4;k<K;k+=128) {
        float values[4];
        #pragma clang loop unroll(full)
        for(uint c=0;c<4;c++) values[c]=x[k+c];
        #pragma clang loop unroll(full)
        for(uint r=0;r<4;r++) {
            #pragma clang loop unroll(full)
            for(uint c=0;c<4;c++) result[r]+=float(weight[(row+r)*K+k+c])*values[c];
        }
    }
    #pragma clang loop unroll(full)
    for(uint r=0;r<4;r++) {
        #pragma clang loop unroll(full)
        for(ushort delta=16;delta>=1;delta>>=1) result[r]+=simd_shuffle_down(result[r],delta);
    }
    if(lane==0) {
        #pragma clang loop unroll(full)
        for(uint r=0;r<4;r++) out[row+r]=HAS_BIAS ? result[r]+float(bias[row+r]) : result[r];
    }
    ''',compile_options={'math_mode':'fast'},
)


def mixed_gemv(x,weight,bias=None):
    n,k=weight.shape
    bm=8 if n>=4096 else 4
    return KERNEL(inputs=[x,weight,bias if bias is not None else weight],
                  template=[('N',n),('K',k),('BM',bm),('HAS_BIAS',bias is not None)],
                  grid=((n//(bm*4))*32,1,bm),threadgroup=(32,1,bm),
                  output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class MixedGEMV(nn.Linear):
    def __call__(self,x):
        if (x.dtype==mx.float32 and self.weight.dtype==mx.bfloat16
                and x.size==self.weight.shape[1]
                and self.weight.shape in {(896,896),(128,896),(4864,896),(896,4864)}):
            return mixed_gemv(x,self.weight,self.get('bias'))
        return super().__call__(x)


def install(model):
    for _,module in model.model.named_modules():
        if type(module) is nn.Linear:module.__class__=MixedGEMV
