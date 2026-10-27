"""Schedule the unchanged stop decision before current-token diffusion.

Derived from the hash-pinned mlx-audio Model.sample (see MLX-AUDIO-LICENSE).
Install before application wrappers so their stop accounting remains intact.
"""
from __future__ import annotations
from types import MethodType
from typing import List, Optional
import mlx.core as mx
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.bailingmm.bailingmm import MING3_MAX_CACHE_LEN, Model


def sample(
    self,
    prompt: str,
    text: str,
    instruction: Optional[str] = None,
    prompt_waveform: Optional[mx.array] = None,
    prompt_text: Optional[str] = None,
    max_decode_steps: int = 200,
    cfg: float = 2.0,
    sigma: float = 0.25,
    temperature: float = 0.0,
    flow_steps: int = 10,
    prompt_latent: Optional[mx.array] = None,
    spk_emb: Optional[List[mx.array]] = None,
):
    if prompt_text is None:
        prompt_latent = None
    elif prompt_latent is None and prompt_waveform is not None:
        prompt_latent = self._encode_prompt_latent(prompt_waveform)

    _, inputs_embeds = self._prepare_input_embed(
        prompt=prompt,
        text=text,
        instruction=instruction,
        prompt_latent=prompt_latent,
        prompt_text=prompt_text,
        spk_emb=spk_emb,
    )

    cache = [KVCache() for _ in range(self.llm_args.num_hidden_layers)]
    latent_history = None

    if self.ming3:
        budget = (MING3_MAX_CACHE_LEN - inputs_embeds.shape[1]) // self.patch_size
        if budget <= 0:
            raise ValueError(
                f"Ming-3.0 prompt has {inputs_embeds.shape[1]} positions and "
                f"exhausts the {MING3_MAX_CACHE_LEN}-position cache. Shorten "
                "the text, instruction, or reference audio."
            )
        max_decode_steps = min(max_decode_steps, budget)

    for step in range(max_decode_steps):
        hidden = self._forward_llm_hidden(inputs_embeds=inputs_embeds, cache=cache)
        z_diff = hidden[:, -1:, :]
        if step == 0:
            latent_history = mx.zeros(
                (1, self.history_patch_size, self.latent_dim),
                dtype=z_diff.dtype,
            )
            if prompt_latent is not None:
                start = self.history_patch_size - prompt_latent.shape[1]
                if start < 0:
                    latent_history = prompt_latent[:, -start:, :]
                else:
                    latent_history = mx.concatenate(
                        [latent_history[:, :start, :], prompt_latent],
                        axis=1,
                    )

        stop_decision = mx.softmax(self.stop_head(z_diff), axis=-1)[0, 0, 1] > 0.5
        mx.async_eval(stop_decision)
        sampled_token_latent, _ = self.flowloss.sample(
            z_diff,
            latent_history,
            cfg=cfg,
            patch_size=self.patch_size,
            sigma=sigma,
            temperature=temperature,
            steps=flow_steps,
        )
        is_last = bool(stop_decision and step > 3)
        yield sampled_token_latent, is_last
        if is_last:
            break

        inputs_embeds = self.linear_proj_audio(sampled_token_latent)
        latent_history = mx.concatenate(
            [latent_history[:, self.patch_size :, :], sampled_token_latent],
            axis=1,
        )


def validate(model):
    current = model.sample
    if not (isinstance(current, MethodType) and current.__self__ is model
            and current.__func__ is Model.sample):
        raise ValueError("Configure Metal before wrapping or replacing model.sample")


def install(model):
    validate(model)
    model.sample = MethodType(sample, model)
