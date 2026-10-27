"""Bound the reference-prompt aggregator's matrices and intermediate lifetime."""
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts.models.bailingmm.bailingmm import DiTBlock
from .aggregator import MixedLinear
from .audio_linear import mixed_linear


class PromptLinear(MixedLinear):
    def __call__(self,x):
        n,k=self.weight.shape
        if (x.dtype==mx.float32 and self.weight.dtype==mx.bfloat16
                and x.size//k>6 and k%64==0 and n%64==0):
            return mixed_linear(x,self.weight,self.get('bias'))
        return super().__call__(x)


class PromptBlock(DiTBlock):
    def __call__(self,x,mask,rope):
        out=super().__call__(x,mask,rope)
        if x.size//x.shape[-1]>6:
            mx.eval(out)
        return out


def install(model):
    for _,module in model.linear_proj_audio.named_modules():
        if isinstance(module,nn.Linear):
            module.__class__=PromptLinear
        elif type(module) is DiTBlock:
            module.__class__=PromptBlock
