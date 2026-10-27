"""Complete each codec transformer block before constructing the next one."""
import mlx.core as mx
from mlx_audio.lm.models.qwen2 import TransformerBlock


class BoundedAudioBlock(TransformerBlock):
    def __call__(self, x, mask=None, cache=None):
        result = super().__call__(x, mask, cache)
        mx.eval(result)
        return result


def install(model):
    for _, module in model.audio.named_modules():
        if type(module) is TransformerBlock:
            module.__class__ = BoundedAudioBlock
