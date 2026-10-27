"""Keep the original RMS reduction, fuse its exact scalar tail without FMA."""
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import RMSNorm

KERNEL=mx.fast.metal_kernel(
    name='ming_exact_rms_tail_v1',input_names=['x','variance','weight'],output_names=['out'],
    header='#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n',
    source='''
    uint i = thread_position_in_grid.x;
    if (i >= SIZE) return;
    uint row = i / DIM;
    float v = variance[row] + 1e-6f;
    float inverse = metal::precise::rsqrt(v);
    float normalized = x[i] * inverse;
    out[i] = normalized * float(weight[i % DIM]);
    ''',compile_options={'math_mode':'safe'},
)


class ExactRMSNorm(RMSNorm):
    def __call__(self,x):
        if x.dtype!=mx.float32 or self.eps!=1e-6:
            return super().__call__(x)
        variance=mx.mean((x.astype(mx.float32))**2,axis=-1,keepdims=True)
        return KERNEL(inputs=[x,variance,self.weight],template=[('SIZE',x.size),('DIM',x.shape[-1])],
                      grid=(x.size,1,1),threadgroup=(256,1,1),
                      output_shapes=[x.shape],output_dtypes=[x.dtype])[0]


def install(model):
    for root in (model.flowloss.cfm.model,model.linear_proj_audio):
        for _,module in root.named_modules():
            if type(module) is RMSNorm:
                module.__class__=ExactRMSNorm
