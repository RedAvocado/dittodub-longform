"""Deterministic solver graph reuse with every RNG draw outside compilation.

Only immutable, synchronous, fixed-shape inference is eligible. The sequence of
ten mx.random.normal calls is unchanged; their results are explicit graph inputs.
"""
import math
from types import MethodType
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import get_epss_timesteps
from .solver_update import update
from .compile_stats import counted_compile


def install(model, compiled=True, chain=False):
    cfm = model.flowloss.cfm
    dit = cfm.model
    original = cfm.sample
    t = get_epss_timesteps(10, dtype=mx.float32)
    t = t + -1.0 * (mx.cos(math.pi / 2 * t) - 1 + t)
    deltas = [t[i + 1] - t[i] for i in range(10)]
    times = [dit.t_embedder(mx.full((2,), t[i], dtype=mx.float32))[:, None, :] for i in range(10)]
    mx.eval(deltas, times)
    probe = mx.zeros((2, 3, 1024), dtype=mx.float32)
    scales_cache = {}
    graph_cache = {}
    stats = {'calls': 0, 'fallbacks': 0, 'traces': 0, 'random_draws': 0,
             'graph_objects': 0, 'compile_traces': 0,
             'traces_semantics': 'legacy graph object count',
             'compile_traces_semantics': 'traced wrapper body entries, including failed attempts'}
    cfm.__dict__['_v8_stats'] = stats

    def execute(y, c, latent_history, noises, scales):
        history = dit.x_embedder(mx.concatenate([latent_history, latent_history], axis=0))
        condition = dit.c_embedder(mx.concatenate([c, mx.zeros_like(c)], axis=0), train=False)
        trajectory = [y]
        for i, (dt, time_embed) in enumerate(zip(deltas, times)):
            current = dit.x_embedder(mx.concatenate([y, y], axis=0))
            full = mx.concatenate([history, current], axis=1)
            full = mx.concatenate([time_embed + condition, full], axis=1)
            if chain:
                from .stack import stack
                full = stack(full, dit, chain=True)
            else:
                for block in dit.blocks:
                    full = block(full, None, dit.rotary_embed)
            prediction = dit.final_layer(full)[:, -1:, :]
            y = update(y, prediction, noises[i], dt, scales[i], 2.0)
            trajectory.append(y)
        return y, trajectory

    def sample(self, noise, c, latent_history, steps=10, cfg_scale=1.0,
               sway_sampling_coef=-1.0, use_epss=True, patch_size=1,
               sigma=0.25, temperature=1.5):
        if not (steps == 10 and cfg_scale == 2.0 and sway_sampling_coef == -1.0
                and use_epss and patch_size == 1 and noise.shape == (1, 64, 1)
                and latent_history.shape == (1, 1, 64) and c.shape == (1, 1, 896)
                and not dit.training
                and all(block.supported(probe, None, dit.rotary_embed) for block in dit.blocks)):
            stats['fallbacks'] += 1
            return original(noise, c, latent_history, steps=steps, cfg_scale=cfg_scale,
                            sway_sampling_coef=sway_sampling_coef, use_epss=use_epss,
                            patch_size=patch_size, sigma=sigma, temperature=temperature)
        key = (sigma, temperature)
        if key not in scales_cache:
            scales_cache.clear()
            scales_cache[key] = [sigma * (temperature ** 0.5) * (mx.abs(dt) ** 0.5) for dt in deltas]
            mx.eval(scales_cache[key])
        scales = scales_cache[key]
        y = noise.transpose(0, 2, 1).astype(mx.float32)
        # No other random operation occurs between these draws in the original
        # fixed inference path. Do not replace them with one batched draw.
        noises = [mx.random.normal(y.shape, dtype=y.dtype) for _ in range(10)]
        stats['random_draws'] += 10
        stats['calls'] += 1
        if not compiled:
            return execute(y, c, latent_history, noises, scales)
        graph_key = (id(dit.rotary_embed), id(dit.rotary_embed.inv_freq),
                     tuple((id(b), b.norm1.eps, b.norm2.eps, b.attn.heads, b.attn.dim_head)
                           for b in dit.blocks))
        if graph_key not in graph_cache:
            # Prime lazy constant caches without consuming any additional RNG.
            execute(y, c, latent_history, noises, scales)
            graph_cache.clear()
            # Capture immutable checkpoint arrays as constants. Passing the
            # entire module as mutable inputs also traces RoPE cache identities,
            # which would try to initialize/evaluate constants inside the trace.
            graph_cache[graph_key] = counted_compile(execute, stats)
            stats['graph_objects'] += 1
            # Backwards-compatible alias; never interpret this as MLX traces.
            stats['traces'] += 1
        return graph_cache[graph_key](y, c, latent_history, noises, scales)

    cfm.sample = MethodType(sample, cfm)
