"""Continuous long-form speech with Ming 3.0 TTS 0.5B on Apple silicon. See core.py for what it changes.

    from dittodub_longform import LongForm
    tts = LongForm()
    audio = tts.speak("A whole chapter ...", voice="US_Emma")          # 44.1 kHz mono float32
    tts.save(audio, "chapter.wav")
    audio = tts.speak(text, reference="me.wav")      # clone; the clip is transcribed automatically
"""
from __future__ import annotations

import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import core, voices
from .core import SAMPLE_RATE, Settings
from .text import normalize

__all__ = ["LongForm", "Settings", "normalize", "SAMPLE_RATE"]
MODEL = "inclusionAI/Ming-3.0-tts-0.5b"
MODEL_REVISION = "8d66b70a00f05ad183fa8a863b4ce609fc83cbc0"  # the exact weights every result was measured with
ASR = "mlx-community/Qwen3-ASR-0.6B-8bit"  # transcribes cloning references; all of Ming's languages
ASR_REVISION = "89e96d92ba34aca20b3e29fb10cc284097d1219f"


def _official_posterior(model):
    """mlx-audio (as of 0.6.2) samples a reference clip's latent with std = exp(logvar / 2); the official AudioVAE
    (diffusers' OobleckDiagonalGaussianDistribution) uses std = softplus(scale) + 1e-4. Only cloning uses this."""
    vae = type(model.audio)

    def encode_latent(self, waveform, waveform_length):
        frame_num = mx.ceil(waveform_length / self.config["enc_kwargs"]["input_dim"]).astype(mx.int32)
        if self.patch_size != -1:
            frame_num = mx.ceil(frame_num / self.patch_size).astype(mx.int32)
        h = self.encoder(waveform).transpose(0, 2, 1)
        mu, scale = mx.split(h, 2, axis=1)
        latent = mu + (nn.softplus(scale) + 1e-4) * mx.random.normal(mu.shape, dtype=mu.dtype)
        return latent.transpose(0, 2, 1), frame_num

    vae.encode_latent = encode_latent


def _read(path):
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{path}: empty or non-finite audio")
    return a.mean(axis=1), sr


_asr = None


def transcribe(audio, sample_rate):
    """Transcript of a reference clip: Qwen3-ASR 0.6B through mlx-audio (about 1 GB, downloaded on first use; the
    language is detected automatically)."""
    global _asr
    from .speaker import resample
    if _asr is None:
        from mlx_audio.stt.utils import load_model as load_asr
        _asr = load_asr(ASR, revision=ASR_REVISION)
    return " ".join(_asr.generate(resample(audio, sample_rate, 16000)).text.split())


def _units(t):
    """Length in roughly English-word units (a Chinese character counts 0.8)."""
    return len(re.findall(r"[A-Za-z0-9']+", t)) + 0.8 * len(re.findall(r"[\u4e00-\u9fff]", t))


def _sentences(t, zh):
    pat = r"(?<=[。！？!?])" if zh else r"(?:(?<=[.!?])|(?<=[.!?][\"”’']))\s+"
    return [x.strip() for x in re.split(pat, t) if x.strip()]


def takes(text: str, language: str = "en", max_words: int = 130) -> list[str]:
    """Split a text into paragraph-sized takes. Ming 3.0 0.5B was trained on short text and cannot keep its place in
    a multi-paragraph prompt, so one take holds up to about one long paragraph (~130 words): whole paragraphs are
    packed together up to that size, and a longer paragraph is split between sentences."""
    zh = language == "zh"
    sep = "" if zh else " "
    out, cur = [], ""
    for para in [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if cur and _units(cur + sep + para) <= max_words:
            cur += sep + para
            continue
        if cur:
            out.append(cur)
            cur = ""
        for sent in _sentences(para, zh) if _units(para) > max_words else [para]:
            if cur and _units(cur + sep + sent) > max_words:
                out.append(cur)
                cur = sent
            else:
                cur = cur + sep + sent if cur else sent
    if cur:
        out.append(cur)
    return out


class LongForm:
    max_words = 130  # per take; see takes()

    def __init__(self, model: str = MODEL, settings: Settings | None = None, fast: bool = True):
        """fast: use the faster Metal kernels (dittodub_longform.accel) when a startup check shows they give the same
        numbers as stock MLX on this Mac; otherwise, or with fast=False, the stock model."""
        from pathlib import Path
        from mlx_audio.tts.utils import load_model
        from mlx_audio.utils import get_model_path
        path = Path(get_model_path(model, revision=MODEL_REVISION if model == MODEL else None))
        self.model, self.fast = load_model(path, strict=True), False
        if fast:
            from . import accel
            self.model, self.fast = accel.enable(self.model, lambda: load_model(path, strict=True))
        _official_posterior(self.model)
        self.settings = settings or Settings()
        self._encoder = None
        self._clones = {}

    def speak(self, text: str, voice: str = "US_Emma", *, reference=None, reference_text: str | None = None,
              language: str | None = None, seed: int = 11, normalize_text: bool = True, verbose: bool = False):
        """Read `text`, one continuous take per paragraph. voice: a built-in voice name; or clone with reference
        (a path or (audio, sample_rate)) and, optionally, its exact transcript in reference_text.
        Returns 44.1 kHz mono float32 audio."""
        language = language or ("zh" if re.search(r"[一-鿿]", text) else "en")
        if normalize_text:
            text = normalize(text, language)
        if reference is not None:
            cond = self._clone(reference, reference_text, verbose)
        else:
            cond = voices.builtin(self.model, voice, language, log=print if verbose else (lambda *_: None))
        parts = takes(text, language, self.max_words)
        started = time.perf_counter()
        with mx.stream(mx.gpu):
            lats = []
            for i, part in enumerate(parts):
                lat, _ = core.generate(self.model, part, instruction=cond.instruction, prompt_latent=cond.prompt_latent,
                                       prompt_text=cond.prompt_text, spk_emb=cond.spk_emb, seed=seed + i,
                                       settings=self.settings, language=language)
                lats.append(lat)
                if verbose and len(parts) > 1:
                    print(f"  take {i + 1}/{len(parts)}: {lat.shape[1] * core.FRAME_SECONDS:.1f} s")
            audio = core.decode(self.model, mx.concatenate(lats, axis=1))  # one latent stream, decoded once
        if verbose:
            took = time.perf_counter() - started
            print(f"{len(audio) / SAMPLE_RATE:.1f} s of audio in {took:.1f} s, {len(parts)} take(s)")
        return audio

    def _clone(self, reference, transcript, verbose=False):
        key = (str(reference) if not isinstance(reference, tuple) else id(reference[0]), transcript)
        if key not in self._clones:
            audio, sr = _read(reference) if not isinstance(reference, tuple) else reference
            if not transcript:
                transcript = transcribe(np.asarray(audio, dtype=np.float32), sr)
                if verbose:
                    print(f"Reference transcript: {transcript}")
            if self._encoder is None:
                from .speaker import SpeakerEncoder
                self._encoder = SpeakerEncoder()
            self._clones[key] = voices.clone(self.model, np.asarray(audio, dtype=np.float32), sr, transcript,
                                             self._encoder)
        return self._clones[key]

    @staticmethod
    def save(audio, path):
        import soundfile as sf
        sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")
