"""Speaker embedding for voice cloning: Ming's CAM++ encoder (MLX weights from mlx-audio) with the official Kaldi
frontend (torchaudio sinc resampling + fbank) re-implemented in NumPy, matching it to float32 precision."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

SPEAKER_REPO = "mlx-community/Step-Audio-2-token2wav"
SPEAKER_REVISION = "e21026b660069e6d1b9ac5c9a51e08ed0201f30e"


def resample(x, orig, new, width=6, rolloff=0.99):
    """torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99)."""
    g = math.gcd(int(orig), int(new))
    orig, new = int(orig) // g, int(new) // g
    x = np.asarray(x, dtype=np.float32)
    if orig == new:
        return x
    base = min(orig, new) * rolloff
    w = math.ceil(width * orig / base)
    idx = np.arange(-w, w + orig, dtype=np.float64) / orig
    t = np.clip((np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx[None]) * base, -width, width)
    win = np.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(t == 0, 1.0, np.sin(t) / t)
    k = (k * win * base / orig).astype(np.float32)
    xp = np.pad(x, (w, w + orig))
    n = (len(xp) - k.shape[1]) // orig + 1
    y = (np.lib.stride_tricks.sliding_window_view(xp, k.shape[1])[::orig][:n] @ k.T).reshape(-1)
    return y[: math.ceil(new * len(x) / orig)]


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def fbank(x, sr=16000, bins=80):
    """torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0) with all other options at their defaults."""
    x = np.asarray(x, dtype=np.float64)
    win, hop, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    n = 1 + (len(x) - win) // hop
    if n < 1:
        raise ValueError("speaker reference is shorter than one 25 ms frame")
    f = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n].copy()
    f -= f.mean(axis=1, keepdims=True)
    f = np.concatenate([f[:, :1] * (1 - 0.97), f[:, 1:] - 0.97 * f[:, :-1]], axis=1)
    f *= (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))) ** 0.85
    power = (np.abs(np.fft.rfft(f, n=n_fft, axis=1)) ** 2).astype(np.float32)
    lo, hi = _mel(20.0), _mel(sr / 2)
    d = (hi - lo) / (bins + 1)
    b = np.arange(bins)[:, None]
    m = _mel(sr / n_fft * np.arange(n_fft // 2))[None]
    banks = np.maximum(0.0, np.minimum((m - (lo + b * d)) / d, ((lo + (b + 2) * d) - m) / d))
    mel = power @ np.pad(banks, ((0, 0), (0, 1))).T.astype(np.float32)
    return np.log(np.maximum(mel, np.finfo(np.float32).eps)).astype(np.float32)


class SpeakerEncoder:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from mlx_audio.codec.models.stepaudio2.speaker import StepAudio2CAMPPlus
        path = hf_hub_download(SPEAKER_REPO, "campplus.safetensors", revision=SPEAKER_REVISION)
        self.model = StepAudio2CAMPPlus()
        self.model.load_weights(list(mx.load(path).items()), strict=True)
        self.model.eval()

    def __call__(self, audio: np.ndarray, sample_rate: int) -> mx.array:
        """Mono float audio -> (1, 192) embedding."""
        wav = resample(audio, sample_rate, 16000)
        if wav.size < 720 or not np.isfinite(wav).all():
            raise ValueError("speaker reference must contain at least 45 ms of finite audio")
        feats = fbank(wav)
        feats -= feats.mean(axis=0, keepdims=True)
        emb = self.model(mx.array(feats[None])).reshape(1, 192)
        mx.eval(emb)
        return emb
