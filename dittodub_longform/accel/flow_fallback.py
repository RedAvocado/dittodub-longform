"""Exact v7 fallback for inputs outside the compiled v8 solver contract."""
import math
from types import MethodType
import mlx.core as mx
from mlx_audio.tts.models.bailingmm.bailingmm import get_epss_timesteps
from .solver_update import update, first_blocks


def install(model):
    """Cache fixed ten-step embeddings and hoist per-token conditioning projections."""
    cfm = model.flowloss.cfm
    dit = cfm.model
    original = cfm.sample
    t = get_epss_timesteps(10, dtype=mx.float32)
    t = t + -1.0 * (mx.cos(math.pi / 2 * t) - 1 + t)
    times = [t[i] for i in range(10)]
    deltas = [t[i+1] - t[i] for i in range(10)]
    embeddings = [dit.t_embedder(mx.full((2,), ti, dtype=mx.float32))[:, None, :] for ti in times]
    mx.eval(times, deltas, embeddings)

    def sample(self, noise, c, latent_history, steps=10, cfg_scale=1.0,
               sway_sampling_coef=-1.0, use_epss=True, patch_size=1,
               sigma=0.25, temperature=1.5):
        if not (steps == 10 and cfg_scale == 2.0 and sway_sampling_coef == -1.0
                and use_epss and patch_size == 1 and noise.shape == (1, 64, 1)
                and latent_history.shape == (1, 1, 64)):
            return original(noise, c, latent_history, steps=steps, cfg_scale=cfg_scale,
                            sway_sampling_coef=sway_sampling_coef, use_epss=use_epss,
                            patch_size=patch_size, sigma=sigma, temperature=temperature)
        history = mx.concatenate([latent_history, latent_history], axis=0)
        condition = mx.concatenate([c, mx.zeros_like(c)], axis=0)
        history_embed = dit.x_embedder(history)
        condition_embed = dit.c_embedder(condition, train=False)
        y = noise.transpose(0, 2, 1).astype(mx.float32)
        trajectory = [y]
        compiled_blocks = []
        for dt, time_embed in zip(deltas, embeddings):
            x = mx.concatenate([y, y], axis=0)
            x_now = dit.x_embedder(x)
            full = mx.concatenate([history_embed, x_now], axis=1)
            prefix = time_embed + condition_embed
            full = mx.concatenate([prefix, full], axis=1)
            if not compiled_blocks:
                full = first_blocks(full, dit, compiled_blocks)
            else:
                for call in compiled_blocks:
                    full = call(full)
            prediction = dit.final_layer(full)[:, -patch_size:, :]
            noise_step = mx.random.normal(y.shape, dtype=y.dtype)
            scale = sigma * (temperature**0.5) * (mx.abs(dt) ** 0.5)
            y = update(y, prediction, noise_step, dt.astype(y.dtype), scale, cfg_scale)
            trajectory.append(y)
        return trajectory[-1], trajectory

    cfm.sample = MethodType(sample, cfm)

