"""Faster Metal kernels for Ming 3.0 0.5B (from DittoDub's ming_accel, v8): fused and cached diffusion-head and
language-model operations that give the same numbers as stock MLX. On an Apple M4 Max with MLX 0.32.4 the output is
bit-identical to stock (every latent of full readings checked) and about 2.4x faster.

enable() checks that claim on the machine it runs on before relying on it: it computes a language-model step, a
flow-head sample and an audio decode on fixed inputs with the stock model, installs the kernels, recomputes them,
and keeps the kernels only if the results agree. Otherwise (or on any error) it returns a freshly loaded stock model.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

TOLERANCE = 1e-2  # relative L2 difference accepted (bf16 rounding is ~3e-3; a wrong kernel is ~1)


def install(model):
    from . import (aggregator, aggregator_block, aggregator_feed_forward, attention, audio_linear, audio_memory,
                   constants, diffusion_linear, feed_forward, flow, flow_fallback, gemv, grouped, normalization,
                   prefill, prefill_memory, promotion, prompt_memory, reduction, residual_norm, rotary, scheduling,
                   stop)
    if not model.ming3 or model.patch_size != 1 or model.training:
        raise ValueError("the kernels cover Ming 3.0 dense inference in eval mode only")
    stop.validate(model)
    constants.install_constants(model)
    rotary.install(model)
    normalization.install(model)
    promotion.install(model, include_aggregator=False)
    gemv.install(model)
    audio_linear.install(model)
    reduction.install(model)
    prefill.install(model)
    audio_memory.install(model)
    aggregator.install(model)
    attention.install(model)
    prompt_memory.install(model)
    diffusion_linear.install(model)
    feed_forward.install(model)
    residual_norm.install(model, compiled=True)
    prefill_memory.install(model)
    scheduling.install(model, "hidden")
    scheduling.install(model, "flow")
    mx.synchronize()
    stop.install(model)
    grouped.install(model)
    flow_fallback.install(model)
    aggregator_block.install(model)
    flow.install(model, compiled=True, chain=True)
    aggregator_feed_forward.install(model)
    mx.synchronize()


def _probe(model):
    """A language-model step, a flow-head sample and an audio decode on fixed inputs (float32 numpy arrays)."""
    from .. import core
    from mlx_audio.lm.models.cache import KVCache
    with mx.stream(mx.gpu):
        _, emb = model._prepare_input_embed(prompt=core.PROMPT, text="A short check of the voice.",
                                            instruction=core.impl.ming3_instruction(ip="US_Emma"),
                                            spk_emb=model._speaker_embeddings(None, True))
        cache = [KVCache() for _ in range(model.llm_args.num_hidden_layers)]
        c = model._forward_llm_hidden(inputs_embeds=emb, cache=cache)[:, -1:, :]
        c2 = model._forward_llm_hidden(inputs_embeds=model.linear_proj_audio(mx.zeros((1, 1, 64))), cache=cache)[:, -1:, :]
        mx.random.seed(7)
        z = core._flow(model, c, mx.zeros((1, 1, 64)), core.Settings())
        lat = mx.array(np.random.default_rng(3).normal(0, 1, (1, 24, 64)).astype(np.float32))
        audio, _, _ = model.audio.decode(lat, use_cache=False, last_chunk=True)
        out = [c, c2, z, audio]
        mx.eval(out)
    return [np.array(x.astype(mx.float32)) for x in out]


def enable(model, reload):
    """Returns (model, True) with the kernels installed and verified, or (stock model, False)."""
    try:
        before = _probe(model)
        install(model)
        after = _probe(model)
        for a, b in zip(before, after):
            if a.shape != b.shape or not np.all(np.isfinite(b)):
                raise ValueError("shape or finiteness mismatch")
            if np.linalg.norm(a - b) > TOLERANCE * np.linalg.norm(a):
                raise ValueError("results differ from stock")
        return model, True
    except Exception:  # noqa: BLE001 - any problem means: use the stock model
        return reload(), False
