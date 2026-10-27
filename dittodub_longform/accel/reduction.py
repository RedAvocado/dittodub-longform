# Native reduction layout adapted from MLX 0.32.4, Copyright Apple Inc.
# Distributed under the MIT license in MLX-LICENSE.
"""Fuse the exact native power, row reduction and RMS scalar operations."""
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import RMSNorm

RMS = mx.fast.metal_kernel(
    name='ming_rms_native_reduction_v3', input_names=['x', 'weight', 'exponent'], output_names=['out'],
    header='#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n',
    source='''
    uint lid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint group = simdgroup_index_in_threadgroup;
    threadgroup float sums[4];
    threadgroup float inverse;
    float total = 0.0f;
    // Match row_reduce_looped: 128 threads, four consecutive reads, two blocks.
    for (uint block = 0; block < 2; block++) {
        for (uint j = 0; j < 4; j++) {
            float value = x[row * 1024 + block * 512 + lid * 4 + j];
            float squared = metal::precise::pow(value, exponent);
            total = squared + total;
        }
    }
    total = 0.0f + total;
    total = simd_sum(total);
    if (lane == 0) sums[group] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = lid < 4 ? sums[lid] : 0.0f;
    value = simd_sum(value);
    if (lid == 0) inverse = metal::precise::rsqrt(value * (1.0f/1024.0f) + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lid; j < 1024; j += 128) {
        float normalized = x[row * 1024 + j] * inverse;
        out[row * 1024 + j] = normalized * float(weight[j]);
    }
    ''', compile_options={'math_mode':'safe'},
)


class FusedRMSNorm(RMSNorm):
    def __call__(self, x):
        # Single-row reductions use a different native launch geometry.
        if (x.dtype != mx.float32 or x.shape[-1] != 1024 or self.eps != 1e-6
                or not 1 < x.size // 1024 < 32):
            return super().__call__(x)
        return RMS(inputs=[x, self.weight, mx.array(2.0, dtype=mx.float32)], grid=(128, x.size // 1024, 1),
                   threadgroup=(128, 1, 1), output_shapes=[x.shape],
                   output_dtypes=[mx.float32])[0]


def install(model):
    for root in (model.flowloss.cfm.model, model.linear_proj_audio):
        for _, module in root.named_modules():
            if isinstance(module, RMSNorm):
                module.__class__ = FusedRMSNorm
