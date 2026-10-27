"""Use native-order tiled FP32 compute for language-model prompt matrices."""
import mlx.core as mx
import mlx.nn as nn
from .gemv import MixedGEMV
from .audio_linear import mixed_linear


class MixedPrefill(MixedGEMV):
    def __call__(self, x):
        n, k = self.weight.shape
        if (x.dtype == mx.float32 and self.weight.dtype == mx.bfloat16
                and 1 < x.size // k
                and (n, k) in {(896, 896), (128, 896), (4864, 896), (896, 4864)}):
            return mixed_linear(x, self.weight, self.get('bias'))
        return super().__call__(x)


def install(model):
    for _, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            module.__class__ = MixedPrefill
