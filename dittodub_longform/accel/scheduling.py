"""Schedule unchanged GPU graphs earlier to overlap graph construction and execution."""
import mlx.core as mx


def install(model, mode):
    if mode == 'hidden':
        original = model._forward_llm_hidden
        def forward(*args, **kwargs):
            value = original(*args, **kwargs)
            mx.async_eval(value)
            return value
        model._forward_llm_hidden = forward
    elif mode == 'flow':
        original = model.flowloss.sample
        def sample(*args, **kwargs):
            values = original(*args, **kwargs)
            mx.async_eval(values[0])
            return values
        model.flowloss.sample = sample
