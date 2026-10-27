"""Bounded CUDA screening of v7's reuse ideas; immutable single-request inference.

This is not a translation of the Metal kernels. Native CUDA operations retain
their arithmetic; CUDA graph replay avoids Python and launch submission work.
Random draws stay in FlowLoss.sample, outside this captured deterministic solver.
"""
import time
import torch


def hoisted_solver(cfm):
    # Resolve after qualification rather than retaining a stale global import.
    from fm.CFM import get_epss_timesteps
    original = cfm.sample
    dit = cfm.model
    cache = {}

    def sample(noise, c, latent_history, sde_noise, cfg_scale=2.0,
               sway_sampling_coef=-1.0, use_epss=True, patch_size=1,
               sigma=0.25, temperature=0.0):
        if (dit.training or noise.shape != (1, 1, 64)
                or latent_history.shape != (1, 1, 64) or c.shape != (1, 1, 896)
                or sde_noise.shape != (10, 1, 1, 64) or cfg_scale != 2.0
                or sway_sampling_coef != -1.0 or not use_epss or patch_size != 1):
            return original(noise=noise, c=c, latent_history=latent_history,
                            sde_noise=sde_noise, cfg_scale=cfg_scale,
                            sway_sampling_coef=sway_sampling_coef, use_epss=use_epss,
                            patch_size=patch_size, sigma=sigma, temperature=temperature)
        key = (noise.device, noise.dtype, torch.is_autocast_enabled('cuda'),
               torch.get_autocast_dtype('cuda'))
        if key not in cache:
            t = get_epss_timesteps(10, noise.device, noise.dtype)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            cache[key] = (
                [t[i + 1] - t[i] for i in range(10)],
                [dit.t_embedder(t[i].repeat(2)).unsqueeze(1) for i in range(10)],
                dit.rotary_embed.forward_from_seq_len(3),
            )
        deltas, times, rope = cache[key]
        history = dit.x_embedder(torch.cat([latent_history, latent_history], dim=0))
        condition = dit.c_embedder(torch.cat([c, torch.zeros_like(c)], dim=0), False)
        y = noise
        for step, (dt, t) in enumerate(zip(deltas, times)):
            x_now = dit.x_embedder(torch.cat([y, y], dim=0))
            x = torch.cat([history, x_now], dim=1)
            x = torch.cat([t + condition, x], dim=1)
            for block in dit.blocks:
                x = block(x, None, rope)
            prediction = dit.final_layer(x)[:, -1:, :]
            pred, null_pred = torch.chunk(prediction, 2, dim=0)
            y = y + (pred + (pred - null_pred) * cfg_scale) * dt
            y = y + sigma * (temperature ** 0.5) * (dt.abs() ** 0.5) * sde_noise[step]
        return y
    return sample


class GraphCall:
    """One exact input signature, fallback otherwise; no RNG or mutable weights."""
    def __init__(self, function):
        self.function = function
        self.signature = None
        self.graph = None
        self.capture_seconds = 0.0
        self.replays = 0

    def __call__(self, *args, **kwargs):
        values = list(args) + list(kwargs.values())
        signature = (len(args), tuple(kwargs), tuple(
            (v.shape, v.dtype, v.device) if torch.is_tensor(v)
            else (type(v), v) for v in values), torch.is_autocast_enabled('cuda'),
            torch.get_autocast_dtype('cuda'), torch.is_inference_mode_enabled())
        if self.signature is not None and signature != self.signature:
            return self.function(*args, **kwargs)
        if self.graph is None:
            started = time.perf_counter()
            self.signature = signature
            # Input views can have different strides after prompt prefill. Copy
            # them into fixed contiguous graph buffers; their source strides
            # do not change the captured computation and must not disable reuse.
            self.values = [v.clone(memory_format=torch.contiguous_format)
                           if torch.is_tensor(v) else v for v in values]
            static_args = self.values[:len(args)]
            static_kwargs = dict(zip(kwargs, self.values[len(args):]))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.function(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self.function(*static_args, **static_kwargs)
            torch.cuda.synchronize()
            self.capture_seconds = time.perf_counter() - started
        for target, value in zip(self.values, values):
            if torch.is_tensor(value):
                target.copy_(value)
        self.graph.replay()
        self.replays += 1
        return self.output.clone()
