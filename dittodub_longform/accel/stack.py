"""Exact chained diffusion blocks, retaining every residual/RMS rounding boundary."""
import math
import mlx.core as mx
from .attention import project_rotary
from .feed_forward import expand,contract_residual
from .residual_norm import dot_partial,residual_norm,ResidualNormBlock


def stack(x,dit,chain=True):
    if not chain:
        for block in dit.blocks:x=ResidualNormBlock.__call__(block,x,None,dit.rotary_embed)
        return x
    normalized=dit.blocks[0].norm1(x)
    for i,block in enumerate(dit.blocks):
        q,k,v=project_rotary(block.attn,normalized,dit.rotary_embed)
        y=mx.fast.scaled_dot_product_attention(q,k,v,scale=1.0/math.sqrt(64))
        y=y.transpose(0,2,1,3).reshape(x.shape)
        layer=block.attn.to_out[0]
        x,normalized=residual_norm(dot_partial(y,layer,2),layer.bias,x,block.norm2.weight,2)
        a,b=block.mlp.ff[0][0],block.mlp.ff[2]
        y=expand(normalized,a.weight,a.bias)
        if i+1<len(dit.blocks):
            x,normalized=residual_norm(dot_partial(y,b,8),b.bias,x,dit.blocks[i+1].norm1.weight,8)
        else:
            x=contract_residual(y,b.weight,b.bias,x)
    return x

