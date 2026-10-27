"""Reuse the exact FP32 casts already performed by MLX's mixed-dtype matmuls."""
import mlx.core as mx
import mlx.nn as nn


class CachedPromotion(nn.Linear):
    def __call__(self, x):
        if x.dtype != mx.float32 or self.weight.dtype != mx.bfloat16:
            return super().__call__(x)
        key = (id(self.weight), id(self.get('bias')))
        cache = self.__dict__.get('_promoted')
        if cache is None or cache[0] != key:
            weight = self.weight.T.astype(mx.float32)
            bias = self['bias'].astype(mx.float32) if 'bias' in self else None
            mx.eval(weight, bias)
            cache = (key, weight, bias)
            self.__dict__['_promoted'] = cache
        _, weight, bias = cache
        return mx.addmm(bias, x, weight) if bias is not None else x @ weight


def install(model, include_aggregator=True):
    # The efficient profile retains only the ten-times-per-token diffusion cache.
    roots = (model.flowloss.cfm.model,)
    if include_aggregator:
        roots += (model.linear_proj_audio,)
    for root in roots:
        for _, module in root.named_modules():
            if type(module) is nn.Linear:
                module.__class__ = CachedPromotion
