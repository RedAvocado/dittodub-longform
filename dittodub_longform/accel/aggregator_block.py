"""Exact two-row audio aggregator, with bounded prompt evaluation."""
import math
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import RotaryEmbedding
from .attention import project_rotary
from .feed_forward import FusedDiffusionBlock, FusedFeedForward
from .rotary import KernelRotary
from .residual_norm import dot_partial, residual_norm


class ResidualNormBlock(FusedDiffusionBlock):
    def supported(self,x,mask,rope):
        if (x.shape!=(1,2,1024) or x.dtype!=mx.float32 or mask is not None
                or type(rope) not in (RotaryEmbedding,KernelRotary) or rope.inv_freq.shape!=(32,)
                or self.attn.heads!=16 or self.attn.dim_head!=64
                or self.norm1.eps!=1e-6 or self.norm2.eps!=1e-6
                or self.norm1.weight.shape!=(1024,) or self.norm2.weight.shape!=(1024,)
                or self.norm2.weight.dtype not in (mx.float32,mx.bfloat16)
                or not self.mlp.supported(x)):
            return False
        return all(layer.weight.dtype==mx.bfloat16 and layer.weight.shape==(1024,1024)
                   and 'bias' in layer and layer.bias.shape==(1024,)
                   for layer in (self.attn.to_q,self.attn.to_k,self.attn.to_v,self.attn.to_out[0]))

    def __call__(self,x,mask,rope):
        # Specialize only the qualified dense diffusion path.
        if not self.supported(x,mask,rope):
            return super().__call__(x,mask,rope)
        q,k,v=project_rotary(self.attn,self.norm1(x),rope)
        y=mx.fast.scaled_dot_product_attention(q,k,v,scale=1.0/math.sqrt(64))
        y=y.transpose(0,2,1,3).reshape(x.shape)
        layer=self.attn.to_out[0]
        p=dot_partial(y,layer,2)
        residual,normalized=residual_norm(p,layer.bias,x,self.norm2.weight,2)
        return self.mlp.with_residual(normalized,residual)


class CompiledResidualNormBlock(ResidualNormBlock):
    def __call__(self,x,mask,rope):
        if not self.supported(x,mask,rope):
            return super().__call__(x,mask,rope)
        key=(id(rope),id(rope.inv_freq),self.norm1.eps,self.norm2.eps,self.attn.heads,self.attn.dim_head)
        cached=self.__dict__.get('_compiled_v6')
        if cached is None or cached[0]!=key:
            # Initialize the small RoPE constants before entering a trace.
            ResidualNormBlock.__call__(self,x,None,rope)
            compiled=mx.compile(lambda value: ResidualNormBlock.__call__(self,value,None,rope),inputs=self)
            cached=(key,compiled)
            self.__dict__['_compiled_v6']=cached
        return cached[1](x)

class AggregatorBlock(CompiledResidualNormBlock):
    def __call__(self,x,mask,rope):
        result=super().__call__(x,mask,rope)
        if x.size//x.shape[-1]>6:
            mx.eval(result)
        return result


def install(model):
    root=model.linear_proj_audio
    root.rotary_embed.__class__=KernelRotary
    for block in root.blocks:
        block.mlp.__class__=FusedFeedForward
        block.__class__=AggregatorBlock
