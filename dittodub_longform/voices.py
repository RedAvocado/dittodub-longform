"""Voices: Ming's built-in voices (each with a primer the model generated itself) and cloned voices (a reference
clip plus its transcript, as in Ming's own cloning)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import core
from .speaker import resample

BUNDLED = Path(__file__).resolve().parent / "primers"
CACHE = Path(os.environ.get("DITTODUB_LONGFORM_CACHE", Path.home() / ".cache" / "dittodub-longform")) / "primers"
PRIMER_TEXT = {"en": "I'll read this passage slowly and clearly, one sentence at a time.",
               "zh": "我会慢慢地、清清楚楚地，一句一句把这段文字读给你听。"}
ENGLISH = ("US_Emma", "US_James", "US_Stella", "UK_George", "UK_Charlotte")


@dataclass
class Conditioning:
    instruction: str | None
    prompt_latent: np.ndarray
    prompt_text: str
    spk_emb: list | None


def builtin(model, voice, language="en", make=True, log=print):
    """Conditioning for a built-in voice; creates and caches its primer on first use."""
    for d in (BUNDLED, CACHE):
        p = d / f"{voice}.npy"
        if p.exists():
            return Conditioning(impl_instruction(voice), np.load(p), json.loads(p.with_suffix(".json").read_text())["text"],
                                model._speaker_embeddings(None, True))
    if not make:
        raise FileNotFoundError(f"no primer for {voice}")
    log(f"Creating a primer for {voice} (one time, about a minute)...")
    from .primer import make_primer
    make_primer(model, voice, language=language, out_dir=CACHE)
    return builtin(model, voice, language, make=False)


def impl_instruction(voice):
    return core.impl.ming3_instruction(ip=voice)


def clone(model, audio, sample_rate, transcript, encoder):
    """Conditioning from a reference clip (mono float array) and its exact transcript."""
    if not transcript or not transcript.strip():
        raise ValueError("cloning needs the exact transcript of the reference clip")
    w44 = resample(audio, sample_rate, core.SAMPLE_RATE)
    mx.random.seed(0)  # the reference's posterior sample is fixed, so every take of a voice sees the same context
    pl = model._encode_prompt_latent(mx.array(w44))
    mx.eval(pl)
    emb = encoder(audio, sample_rate)
    return Conditioning(None, np.array(pl[0].astype(mx.float32)), " ".join(transcript.split()),
                        model._speaker_embeddings(emb, False))
