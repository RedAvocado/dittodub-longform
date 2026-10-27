# DittoDub Long-Form

**Whole passages in one continuous take with [Ming 3.0 TTS 0.5B](https://github.com/inclusionAI/Ming-3.0-tts), on Apple silicon.**
By [DittoDub](https://dittodub.com).

Ming 3.0 TTS reads short text beautifully. Give it a paragraph in one go and, past half a minute or so, it tends to
lose its place (skipping ahead or re-reading a sentence), the voice slowly changes colour (hollow, boomy, hissy: it
sounds "muffled" by the end), and some words crack. The usual workaround is to cut the text into pieces of a sentence
or two and synthesize them separately, so every piece starts cold.

This package reads each paragraph (up to about 130 words) as **one generation with full context**, and a longer text
as consecutive paragraph-sized takes in the same voice, decoded as one continuous audio stream. No audio
post-processing. It uses the stock inclusionAI weights; every change is in how the model is run (see
[How it works](#how-it-works)).

> Status: research release. Measured on Apple M4 Max with MLX. English and Mandarin.

## Install

Requires an Apple silicon Mac and Python 3.10+.

```sh
pip install git+https://github.com/RedAvocado/dittodub-longform
```

The model (about 2.8 GB) downloads from Hugging Face on first use.

## Use

```sh
dittodub-longform "A short passage to read." -o speech.wav
dittodub-longform --file chapter.txt -o chapter.wav --voice UK_George
cat chapter.txt | dittodub-longform -o chapter.wav

# clone a voice from a 5-15 s clip of one speaker (it is transcribed automatically)
dittodub-longform --file chapter.txt -o chapter.wav --reference me.wav
```

```python
from dittodub_longform import LongForm

tts = LongForm()
audio = tts.speak(open("chapter.txt").read(), voice="US_Emma")   # 44.1 kHz mono float32
tts.save(audio, "chapter.wav")
audio = tts.speak("Read this in my voice.", reference="me.wav")
```

Voices: `US_Emma` (default), `US_James`, `US_Stella`, `UK_George`, `UK_Charlotte`, and Ming's 100+ Chinese voices (for
example `灵小柔`, `灵岩`). A Chinese voice without a bundled primer gets one made on first use (about a minute, then
cached). Separate paragraphs with a blank line; another `--seed` gives another reading of the same text.

Numbers, money, dates, times, units and common abbreviations are spelled out before synthesis ("$3.50" ->
"three dollars and fifty cents"); turn this off with `--no-normalize`.

## Results

Held-out test: six English passages (news, dialogue, science, a letter, a recipe, an essay; 96-139 words each) and a
Mandarin one, two seeds, five built-in English voices (US_Stella was never used while developing the method), two
cloned voices and two Mandarin voices: 88 takes. Each passage is one continuous take (30-55 s), compared with the
same model reading it in one take without these changes.

| | Ming 3.0, one take | This package |
|---|---|---|
| Word error rate | 10.4% | 1.4% |
| Takes with skipped or repeated words | 35 of 88 | 1 of 88 |
| Timbre drift in the last third (first third) | 5.8 dB (4.0) | 3.0 dB (2.6) |
| Built-in English voices: word error rate, failed takes | 7.3%, 20 of 60 | 1.0%, 0 of 60 |
| Cloned voices (8 s references) | 19.8%, 15 of 24 | 2.1%, 1 of 24 |

Voice similarity (speaker-embedding cosine, CAM++) of the first and last third of each take against the same
voice's own single sentences, and for cloned voices against the reference clip:

| | Ming 3.0, one take | This package |
|---|---|---|
| Built-in English voices, start / end | 0.868 / 0.807 | 0.928 / 0.914 |
| Cloned voices vs reference, start / end | 0.908 / 0.878 | 0.896 / 0.884 |

(Two fresh sentence sets of the same built-in voice score 0.92-0.98 against each other.) Delivery against the
voice's own single sentences, built-in voices: pitch range 3.49 vs 3.93 semitones (Ming one take: 3.25), pitch level
-0.55 semitones (Ming: -1.37), 3.6 words per second (Ming: 3.9; single sentences 3.1).

Four more cloned voices (LibriTTS speakers, 9-11 s references, 48 takes): 1.2% word error rate, 1 failed take.

Sudden pitch breaks ("cracks"): in short single sentences (230 sentences, five voices) 1.36 per sentence with
Ming's default settings, 0.48 with this package's guidance; in the long takes 8.5 per voiced minute for the built-in
voices (Ming one take 8.6) and 12.2 for cloned voices (Ming 14.4).

Longer texts: the six passages as one 3.5-4 minute reading (one take per paragraph). Built-in voices (all five):
0.7-1.8% word error rate, one 4-word skip in five readings, and the timbre stays put (drift in the last third within
0.5 dB of the first third). The cloned voice (Marina) had three short skips (10% word error rate).

- *Content failures*: takes where a run of four or more words was skipped or repeated, or the ending was missing
  (word alignment of a Whisper transcript against the text).
- *Timbre drift*: per sentence, the 1/3-octave spectrum of the long take against a fresh single-sentence take of the
  same sentence by the same voice, in dB (fresh takes of the same sentence differ from each other by about 2 dB).

## How it works

1. **Primer.** The context starts from a short clip of the same voice that the model generated itself, passed in
   through Ming's own voice-reference input. Long prompts can otherwise push a built-in voice into a narrow-band
   (~10 kHz) "recording mode" within the first second or two, and the model keeps whatever bandwidth its own
   history has. The primer is context only; it is not in the output. For a cloned voice, the reference clip plays
   this role.
2. **RoPE scaling.** The language model was trained on sequences up to 512 positions; a long take runs to several
   thousand. NTK-aware scaling of the rotary position base keeps long distances in the trained range.
3. **Reading-position guard.** Two attention heads in the language model (layer 10, heads 12 and 13) follow where
   it is reading. Text more than a few tokens behind that point is masked, so it cannot jump back and re-read.
   Late in a long take the model is drawn towards the end of the passage (it was trained on short clips, where a
   long stretch of audio means the text is nearly done); when the heads suddenly land far ahead, the text beyond
   the reading point is hidden for a moment and the model is not allowed to stop, until reading settles.
4. **Guidance on the early flow steps only.** Each 80 ms audio frame is produced by ten flow-matching steps with
   classifier-free guidance. Guidance on the late, fine-detail steps compounds over a long take into a slow drift
   of the voice's timbre. Guiding only the first six steps keeps the articulation and removes most of the drift.

## Limits

- Apple silicon only (MLX). Measured on an M4 Max: about 0.65x real time (a 4-minute reading takes about 2.5
  minutes).
- One take covers about one long paragraph. Ming 3.0 0.5B was trained on short text and loses its place in a
  multi-paragraph prompt, so longer texts are read paragraph by paragraph; each paragraph starts from the same voice
  primer, so the voice stays the same, but delivery does not carry across paragraph boundaries.
- Tested in English and Mandarin. Ming's other languages (ja, ko, de, fr, es) run but are untested here.
- Cloned voices are less robust than built-in ones, and a reference clip with noise or several speakers will carry
  that into the reading.
- US_Stella reads a little brighter (about +5 dB at 10-13 kHz) than her own single sentences.

## Credits

- [Ming 3.0 TTS](https://github.com/inclusionAI/Ming-3.0-tts) by inclusionAI (MIT): the model and weights.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) by Prince Canuma (MIT): the MLX port this builds on.
- [Whisper](https://github.com/openai/whisper) (MIT) via [mlx-whisper](https://github.com/ml-explore/mlx-examples) transcribes cloning references.
- NTK-aware RoPE scaling (bloc97, 2023); guidance on a limited interval of the sampling steps
  (Kynkäänniemi et al., NeurIPS 2024).

This package also fixes one small port difference: mlx-audio samples a cloning reference's latent with the wrong
noise scale (`exp(logvar / 2)` instead of the official `softplus(scale) + 1e-4`).

Use voice cloning only with the speaker's consent.

## License

MIT. The Ming 3.0 weights are released by inclusionAI under the MIT license.
