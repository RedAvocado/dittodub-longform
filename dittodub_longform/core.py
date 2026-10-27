"""One continuous generation for a whole passage with Ming 3.0 TTS 0.5B on Apple silicon (MLX).

No text splitting, no context resets, no stitching, no audio post-processing. Stock inclusionAI weights; every
change is in how the model is run:

1. Primer. The context starts from a short clip of the same voice that the model generated itself, passed in
   through Ming's own voice-reference input (prompt_latent + prompt_text). Long prompts can otherwise push a
   built-in voice into a narrow-band (~10 kHz) "recording mode" in the first second or two, and the model keeps
   whatever bandwidth its own history has. The primer is context only; it is not in the output. Cloned voices
   use the reference clip in the same place.
2. NTK RoPE scaling of the LLM (base x 1.5^(64/62)), so text distances stay in the trained range past the 512
   positions the model was trained on.
3. Text-position guard. Two attention heads (LLM layer 10, heads 12/13) follow the reading position. Text more than
   a few tokens behind a slowly advancing frontier is masked (no jumping back to re-read a sentence). When the heads
   suddenly land far ahead (the start of a skip, usually towards the end of the passage), text beyond the frontier
   is hidden for a moment until reading settles.
4. Guidance only on the early flow steps. Classifier-free guidance on the late (fine-detail) steps of each frame
   compounds over a long take into a slow timbre drift (hollow 1 kHz, boom, hiss; "muffled at the end"). Guiding
   the first steps only keeps the articulation and removes most of the drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_audio.tts.models.bailingmm.bailingmm as impl
from mlx_audio.lm.models.cache import KVCache

PROMPT = "Please generate speech based on the following description.\n"
ALIGN_LAYER, ALIGN_HEADS = 10, (13, 12)
WIN, START = 5, 8
SAMPLE_RATE = 44100
FRAME_SECONDS = 0.08


@dataclass(frozen=True)
class Settings:
    cfg: float = 2.0            # classifier-free guidance strength ...
    guided_steps: int = 6       # ... on the first `guided_steps` of the `flow_steps` solver steps, none after
    flow_steps: int = 10
    sigma: float = 0.25
    temperature: float = 0.0
    ntk: float = 1.5            # RoPE base scale for the LLM
    margin: int = 6             # mask target text more than `margin` tokens behind the frontier
    rate: float = 0.5           # max frontier advance per 80 ms frame (tokens)
    jump: int | None = 14       # skip guard: heads landing more than `jump` tokens past the frontier ...
    ahead: int = 10             # ... hide text more than `ahead` tokens past it ...
    hold: int = 25              # ... for this many frames (no stopping meanwhile: the end of the text is hidden)
    rewind: int = 6             # ... after regenerating the last `rewind` frames, so the skip never reaches the audio
    max_vetoes: int = 60
    max_positions: int = 4096   # context budget (prompt + text + audio frames)


class _GuardedAttention(impl.MingQwen2Attention):
    """Native LLM attention plus the guard's additive text mask and the alignment readout (single-token steps)."""
    _guard = None
    _layer_idx = -1

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        g = self._guard
        if L == 1 and g is not None:
            if g.bias is not None:
                b = g.bias[..., :k.shape[2]]
                mask = b if mask is None else mask + b
            if self._layer_idx == ALIGN_LAYER:
                rep = self.n_heads // self.n_kv_heads
                heads = list(ALIGN_HEADS)
                kk = mx.repeat(k, rep, axis=1)[:, heads]
                s = (q[:, heads] * self.scale) @ kk.transpose(0, 1, 3, 2)
                if mask is not None and not isinstance(mask, str):
                    s = s + mask
                w = mx.softmax(s.astype(mx.float32), axis=-1)[0, :, 0, g.t0:g.t1].sum(0)
                g.pending = mx.argmax(w)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class _TextGuard:
    """Reading-position tracker and text mask (see module docstring, item 3)."""

    def __init__(self, t0, t1, s: Settings):
        self.t0, self.t1, self.s = t0, t1, s
        self.bias = self.pending = self.key = None
        self.trace, self.frontier, self.hold_until = [], 0.0, -1
        self.log, self.vetoes, self.rewinds = [], 0, 0

    def veto_stop(self, step):
        """Refuse a stop while the skip guard hides the end of the text (it would be premature)."""
        if step < self.hold_until and self.vetoes < self.s.max_vetoes:
            self.vetoes += 1
            return True
        return False

    def update(self, step):
        """Track the reading position after frame `step`; returns how many frames to regenerate (0 = none)."""
        if self.pending is None:
            return 0
        s = self.s
        am = int(self.pending.item())
        self.pending = None
        rewound = False
        if s.jump is not None and step >= START and am > self.frontier + s.jump:
            if s.rewind and step >= 1 and self.rewinds < 30:
                step = max(1, step - s.rewind + 1)
                del self.trace[step:], self.log[step:]
                self.rewinds, rewound = self.rewinds + 1, True
            else:
                self.trace.append(int(self.frontier))  # a landing far ahead must not move the frontier
            self.hold_until = step + s.hold
        else:
            self.trace.append(am)
        if len(self.trace) >= WIN and not rewound:
            target = min(float(np.median(self.trace[-WIN:])), self.frontier + s.rate)
            self.frontier = max(self.frontier, target)
        cut = int(self.frontier) - s.margin
        fwd = int(self.frontier) + s.ahead if step < self.hold_until else None
        key = (max(cut, 0), fwd)
        if step >= START and (cut > 0 or fwd is not None or self.key is not None) and key != self.key:
            b = np.zeros((1, 1, 1, s.max_positions + 64), np.float32)
            if cut > 0:
                b[..., self.t0:self.t0 + cut] = -1e9
            if fwd is not None and self.t0 + fwd < self.t1:
                b[..., self.t0 + fwd:self.t1] = -1e9
            self.bias, self.key = mx.array(b).astype(mx.bfloat16), key
        if rewound:
            return s.rewind
        self.log.append((step, am, self.frontier))
        return 0


def _times(steps):
    t = impl.get_epss_timesteps(steps, dtype=mx.float32)
    return t + -1.0 * (mx.cos(math.pi / 2 * t) - 1 + t)


def _velocity(cfm, x, t, c, hist, cfg):
    if abs(cfg) < 1e-5:
        pred = cfm.model(x=x, t=mx.full((x.shape[0],), t, dtype=x.dtype), c=c, latent_history=hist)
        return pred[:, -1:, :].astype(mx.float32)
    # forward_with_cfg batches the unconditional branch only when cfg_scale != 1 (the value is not used there)
    out = cfm.model.forward_with_cfg(x=x, t=t, c=c, latent_history=hist, cfg_scale=cfg if cfg != 1.0 else 2.0,
                                     patch_size=1)
    pred, null = mx.split(out, 2, axis=0)
    return (pred + (pred - null) * cfg).astype(mx.float32)


def _flow(model, c, hist, s: Settings):
    """Ming's flow-matching solver (same random draws as FlowLoss.sample), guidance on the early steps only."""
    y = mx.random.normal((1, 64, 1), dtype=mx.float32).transpose(0, 2, 1).astype(mx.float32)
    t = _times(s.flow_steps)
    for i in range(1, t.shape[0]):
        dt = t[i] - t[i - 1]
        y = y + dt.astype(y.dtype) * _velocity(model.flowloss.cfm, y, t[i - 1], c, hist,
                                               s.cfg if i <= s.guided_steps else 0.0)
        n = mx.random.normal(y.shape, dtype=y.dtype)
        y = y + s.sigma * (s.temperature ** 0.5) * (mx.abs(dt) ** 0.5) * n
    return y


def _text_span(model, ids, text):
    ids, tt = list(np.array(ids[0])), model._encode(text)
    for i in range(len(ids) - len(tt), -1, -1):
        if ids[i:i + len(tt)] == tt:
            return i, i + len(tt)
    raise ValueError("target text tokens not found in the prompt")


def prompt(model, text, *, instruction, prompt_latent, prompt_text, spk_emb, language="en"):
    p = PROMPT
    lang = impl.MING3_TARGET_LANGUAGES.get(language)
    if lang:
        p += f"Target language: {lang}.\n"
    text = impl._add_period(" ".join(text.split()))
    pl = mx.array(np.asarray(prompt_latent, dtype=np.float32)[None])
    ids, emb = model._prepare_input_embed(prompt=p, text=text, instruction=instruction, prompt_latent=pl,
                                          prompt_text=prompt_text, spk_emb=spk_emb)
    return ids, emb, text, pl


def generate(model, text, *, instruction, prompt_latent, prompt_text, spk_emb, seed, settings: Settings,
             language="en", on_frame=None):
    """Latents (1, T, 64) for the whole text in one continuous take. Raises if the budget runs out first."""
    s = settings
    ids, emb, text, pl = prompt(model, text, instruction=instruction, prompt_latent=prompt_latent,
                                prompt_text=prompt_text, spk_emb=spk_emb, language=language)
    if emb.shape[1] >= s.max_positions:
        raise ValueError("text too long for one take (prompt alone exceeds the position budget)")
    guard = _TextGuard(*_text_span(model, ids, text), s)
    layers = model.model.model.layers
    saved = [(l.self_attn.__class__, l.self_attn.rope) for l in layers]
    for i, l in enumerate(layers):
        l.self_attn.__class__ = _GuardedAttention
        l.self_attn._layer_idx, l.self_attn._guard = i, guard
        l.self_attn.rope = nn.RoPE(64, traditional=False, base=1e6 * s.ntk ** (64 / 62))
    try:
        cache = [KVCache() for _ in range(model.llm_args.num_hidden_layers)]
        mx.random.seed(seed)
        hist = pl[:, -1:, :]
        latents, stopped = [], False
        budget, step, iters = s.max_positions - emb.shape[1], 0, 0
        while step < budget and iters < budget + 400:
            iters += 1
            c = model._forward_llm_hidden(inputs_embeds=emb, cache=cache)[:, -1:, :]
            z = _flow(model, c, hist, s)
            stop = mx.softmax(model.stop_head(c), axis=-1)[0, 0, 1]
            mx.eval(z, stop)
            latents.append(z)
            rewind = guard.update(step)
            if rewind:  # drop the last frames from the output, the KV cache and the history, and redo them
                r = max(1, step - rewind + 1)
                for cc in cache:
                    cc.trim(step - r + 1)
                del latents[r:]
                emb, hist, step = model.linear_proj_audio(latents[r - 1]), latents[r - 1], r
                continue
            if on_frame:
                on_frame(step)
            if bool(stop > 0.5 and step > 3) and not guard.veto_stop(step):
                stopped = True
                break
            emb = model.linear_proj_audio(z)
            hist = z
            step += 1
    finally:
        for l, (cls, rope) in zip(layers, saved):
            l.self_attn.__class__, l.self_attn.rope = cls, rope
            for a in ("_guard", "_layer_idx"):
                l.self_attn.__dict__.pop(a, None)
    if not stopped:
        raise RuntimeError(f"the take used the whole position budget ({s.max_positions}) without finishing")
    return mx.concatenate(latents, axis=1), guard.log


def decode(model, latents):
    raw, _, _ = model.audio.decode(latents, use_cache=False, last_chunk=True)
    audio = impl._trim_trailing_silence(raw[0], model.sample_rate)
    mx.eval(audio)
    return np.array(audio.astype(mx.float32))
