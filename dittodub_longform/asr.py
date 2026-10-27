"""Transcripts of cloning references: Qwen3-ASR 0.6B through mlx-audio, in less memory, with the same bytes.

Stock mlx-audio builds the whole audio encoder, and the whole prompt prefill, as one graph before running it, so the
intermediate buffers of every layer are alive at once. Here each layer is finished before the next is built: the same
operations on the same shapes in the same order, so the encoder output, the log-probabilities at every decoding step
and the transcript are byte-for-byte those of stock, while a 10 s clip peaks at 1.14 GB of Metal memory instead of
1.79 GB. The model is loaded for each new reference and freed straight after, so it holds nothing while Ming speaks.
"""
import gc

import mlx.core as mx
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import AudioEncoderLayer, TextDecoderLayer

MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"  # all of Ming's languages, detected automatically
REVISION = "89e96d92ba34aca20b3e29fb10cc284097d1219f"


class _AudioLayer(AudioEncoderLayer):
    def __call__(self, hidden_states, mask=None):
        out = super().__call__(hidden_states, mask)
        mx.eval(out)
        return out


class _PrefillLayer(TextDecoderLayer):
    def __call__(self, hidden_states, mask=None, cache=None):
        out = super().__call__(hidden_states, mask, cache)
        if out.shape[1] > 1:  # one-token decoding steps are left alone
            mx.eval(out)
        return out


def layer_by_layer(model):
    """Finish the audio encoder and the prompt prefill one layer at a time (in place)."""
    root = getattr(model, "_model", model)
    for layer in root.audio_tower.layers:
        if type(layer) is AudioEncoderLayer:
            layer.__class__ = _AudioLayer
    # Prefill keeps only the cache, so the last layer stays lazy and its unused output is never computed.
    for layer in root.model.layers[:-1]:
        if type(layer) is TextDecoderLayer:
            layer.__class__ = _PrefillLayer
    return model


def transcribe(audio, sample_rate):
    """Transcript of a mono float clip (the model, about 1 GB, downloads on first use)."""
    from mlx_audio.stt.utils import load_model

    from .speaker import resample
    model = layer_by_layer(load_model(MODEL, revision=REVISION))
    try:
        text = model.generate(resample(audio, sample_rate, 16000)).text
    finally:
        del model
        gc.collect()
        mx.clear_cache()
    return " ".join(text.split())
