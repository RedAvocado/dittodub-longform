"""Fuse rotary elementwise work with explicit intermediate rounding and no FMA."""
import mlx.core as mx
from .constants import CachedRotary

KERNEL = mx.fast.metal_kernel(
    name='ming_exact_rotary_pairs_v1',
    input_names=['q','k','cosine','sine'], output_names=['oq','ok'],
    header='#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n',
    source='''
    uint p = thread_position_in_grid.x;
    if (p >= PAIRS) return;
    uint i = p * 2;
    uint phase = p % PHASES;
    float c = cosine[phase], s = sine[phase];
    float qe = q[i], qo = q[i+1], ke = k[i], ko = k[i+1];
    float qec = qe*c, qos = qo*s, qes = qe*s, qoc = qo*c;
    float kec = ke*c, kos = ko*s, kes = ke*s, koc = ko*c;
    oq[i] = qec-qos; oq[i+1] = qes+qoc;
    ok[i] = kec-kos; ok[i+1] = kes+koc;
    ''', compile_options={'math_mode':'safe'},
)


class KernelRotary(CachedRotary):
    def __call__(self, q, k, offset=0):
        if q.dtype != mx.float32 or k.dtype != mx.float32 or q.shape != k.shape:
            return super().__call__(q,k,offset)
        key = (q.shape[2], offset, id(self.inv_freq))
        cache = self.__dict__.setdefault('_constant_cache',{})
        if key not in cache:
            pos = mx.arange(offset,offset+q.shape[2],dtype=mx.float32)
            freqs = mx.outer(pos,self.inv_freq.astype(mx.float32))
            cosine,sine=mx.cos(freqs)[None,None,:,:],mx.sin(freqs)[None,None,:,:]
            mx.eval(cosine,sine)
            if len(cache)>=8:cache.clear()
            cache[key]=(cosine,sine)
        cosine,sine=cache[key]
        return tuple(KERNEL(
            inputs=[q,k,cosine,sine],
            template=[('PAIRS',q.size//2),('PHASES',cosine.size)],
            grid=(q.size//2,1,1),threadgroup=(128,1,1),
            output_shapes=[q.shape,k.shape],output_dtypes=[q.dtype,k.dtype],
        ))


def install(model):
    for rope in (model.flowloss.cfm.model.rotary_embed,model.linear_proj_audio.rotary_embed):
        rope.__class__=KernelRotary
