"""Bound language-model prompt intermediates, retaining the native KV cache."""
import mlx.core as mx
from mlx_audio.lm.models.qwen2 import TransformerBlock


class BoundedPrefillBlock(TransformerBlock):
    def __call__(self,x,mask=None,cache=None):
        out=super().__call__(x,mask,cache)
        if x.shape[1]>1:
            mx.eval(out)
        return out


def install(model):
    for block in model.model.model.layers:
        assert type(block) is TransformerBlock
        block.__class__=BoundedPrefillBlock
