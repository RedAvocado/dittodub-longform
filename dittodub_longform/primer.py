"""Primer creation for a built-in voice (bundled primers were made the same way).

The take inherits the primer's colour and pace, and a primer's own spectrum predicts that poorly, so each candidate
(a short model-generated clip of the primer sentence, several seeds) is tested directly: continue a three-sentence
calibration passage from it, compare that continuation's spectrum with fresh single-sentence takes of the same
sentences, and keep the closest, with a mild penalty for a primer faster than about 3 words per second.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import core

PRIMER_CFG = 0.5
SEEDS = (1424039532, 7, 99, 424242, 1, 2)
REFERENCE = {
    "en": ("The library opened early on Monday, and a few students were already waiting outside.",
           "She folded the map, checked the weather, and decided to take the longer road home.",
           "Most of the town still remembers the winter when the river froze for three weeks."),
    "zh": ("图书馆星期一很早就开门了，门口已经有几个学生在等着。",
           "她把地图折好，看了看天气，决定走那条远一点的路回家。",
           "镇上的人到现在还记得，那年冬天河水结了整整三个星期的冰。"),
}
_BANDS = ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 10000), (10000, 12000), (12000, 16000), (16000, 20000))


def _speech(a):
    """First and last 80 ms frame within 40 dB of the loudest frame."""
    e = 10 * np.log10(np.square(a[: len(a) // 3528 * 3528].reshape(-1, 3528)).mean(1) + 1e-12)
    idx = np.nonzero(e > e.max() - 40)[0]
    return int(idx[0]), int(idx[-1])


def _short_take(model, voice, text, seed):
    """Native single short utterance -> (latents trimmed to the speech end + ~0.3 s, audio of the speech)."""
    mx.random.seed(seed)
    lat = [z for z, _ in model.sample(
        prompt=core.PROMPT, text=text, instruction=core.impl.ming3_instruction(ip=voice), max_decode_steps=400,
        cfg=PRIMER_CFG, sigma=0.25, temperature=0.0, flow_steps=10, spk_emb=model._speaker_embeddings(None, True))]
    lat = np.array(mx.concatenate(lat, axis=1).astype(mx.float32))[0]
    raw, _, _ = model.audio.decode(mx.array(lat[None]), use_cache=False, last_chunk=True)
    a = np.array(raw[0].astype(mx.float32))
    s0, s1 = _speech(a)
    return lat[: min(len(lat), s1 + 1 + 4)], a[s0 * 3528:(s1 + 1) * 3528]


def _tilt(audio):
    """Band levels relative to 80-4000 Hz (dB), power-averaged over 2048-sample frames."""
    n = len(audio) // 2048 * 2048
    P = (np.abs(np.fft.rfft(audio[:n].reshape(-1, 2048) * np.hanning(2048), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(2048, 1 / 44100)
    ref = P[(f >= 80) & (f < 4000)].sum()
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].sum() / ref) for lo, hi in _BANDS])


def make_primer(model, voice, language="en", out_dir=None, settings=None, target_wps=3.0):
    from .voices import PRIMER_TEXT
    settings = settings or core.Settings()
    lang = "zh" if language == "zh" else "en"
    primer_text, refs = PRIMER_TEXT[lang], REFERENCE[lang]
    target = _tilt(np.concatenate([_short_take(model, voice, t, SEEDS[0])[1] for t in refs]))
    calib = "".join(refs) if lang == "zh" else " ".join(refs)
    words = sum(1 for c in primer_text if "一" <= c <= "鿿") / 1.5 if lang == "zh" else len(primer_text.split())
    best, tried = None, []
    for seed in SEEDS:
        lat, speech = _short_take(model, voice, primer_text, seed)
        wps = words / (len(speech) / 44100)
        try:
            clat, _ = core.generate(model, calib, instruction=core.impl.ming3_instruction(ip=voice), prompt_latent=lat,
                                    prompt_text=primer_text, spk_emb=model._speaker_embeddings(None, True),
                                    seed=SEEDS[0], settings=settings, language=language)
        except RuntimeError:
            continue
        raw, _, _ = model.audio.decode(clat, use_cache=False, last_chunk=True)
        a = np.array(raw[0].astype(mx.float32))
        s0, s1 = _speech(a)
        dev = _tilt(a[s0 * 3528:(s1 + 1) * 3528]) - target
        score = float(np.mean(np.abs(dev))) + 2.0 * max(0.0, wps - target_wps)
        tried.append(dict(seed=seed, score=round(score, 2), wps=round(wps, 2)))
        if best is None or score < best[0]:
            best = (score, seed, lat, dev, wps)
    if best is None:
        raise RuntimeError(f"no usable primer for {voice}")
    score, seed, lat, dev, wps = best
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{voice}.npy", lat)
    (out / f"{voice}.json").write_text(json.dumps(
        dict(text=primer_text, source=f"model-generated, cfg {PRIMER_CFG}, seed {seed}, calibrated", frames=len(lat),
             calibration_band_deviation_db=[round(float(d), 1) for d in dev], words_per_second=round(wps, 2),
             candidates=tried), indent=1, ensure_ascii=False))
    return out / f"{voice}.npy"
