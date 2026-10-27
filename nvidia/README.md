# DittoDub NVIDIA accelerator for Ming 3.0 TTS

By [DittoDub](https://dittodub.com). Speeds up [Ming 3.0 TTS 0.5B](https://github.com/inclusionAI/Ming-3.0-tts)'s
official PyTorch inference on NVIDIA GPUs with **identical output**: the same audio, latents and random-number
state as stock, about twice as fast.

| | Stock Ming 3.0 | With this package |
|---|---|---|
| Real-time factor, RTX 5090 (lower is faster) | 0.340 | 0.165 (2.06x) |
| Output | | bit-identical audio, latents and RNG state |

It reuses conditioning and constants across the ten flow steps, captures the flow solver in CUDA graphs (random draws
stay outside), fuses RMS-norm pointwise work after the native reduction, and skips vocabulary logits the speech
generator never reads. Native matrix multiplies and attention are unchanged.

This accelerates Ming's standard short-text generation. The long-form method in this repository runs on Apple
silicon (MLX); see the [main README](../README.md).

## Requirements

It switches on only for the exact runtime its equality was proven on, and refuses (with the reason) otherwise:
NVIDIA GeForce RTX 5090, CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 4.46.3, x-transformers 2.9.2, diffusers
0.33.0, Triton 3.6.0, and the official Ming 3.0 code at commit
`028c1eace193e7acdb256453da89b6e9207e052b` with the original BF16 weights.

```sh
pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0' 'torchaudio==2.11.0'
pip install 'transformers==4.46.3' 'x-transformers==2.9.2' 'diffusers==0.33.0' 'triton==3.6.0'
pip install "git+https://github.com/RedAvocado/dittodub-longform#subdirectory=nvidia"
git clone https://github.com/inclusionAI/Ming-3.0-tts && git -C Ming-3.0-tts checkout 028c1eace193e7acdb256453da89b6e9207e052b
export PYTHONPATH=$PWD/Ming-3.0-tts
```

## Use

```python
import torch
from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from transformers import AutoTokenizer
from dittodub_nvidia import configure

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    "inclusionAI/Ming-3.0-tts-0.5b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
).eval().to(torch.bfloat16).to("cuda")
model.tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ming-3.0-tts-0.5b", trust_remote_code=True)
configure(model)        # then generate as usual; warm up once before timing (CUDA graphs are captured lazily)
```

Do not train, move or modify the configured model, or use it from several threads at once; load a fresh model to go
back to stock.

MIT license. Parts follow the official Ming 3.0 code (MIT, see `Ming-LICENSE`).
