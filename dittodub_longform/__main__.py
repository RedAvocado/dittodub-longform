"""Read text aloud with Ming 3.0 TTS, one continuous take per paragraph.

  dittodub-longform "Your passage." -o speech.wav
  dittodub-longform --file chapter.txt -o chapter.wav --voice UK_George
  cat chapter.txt | dittodub-longform -o chapter.wav
  dittodub-longform --file chapter.txt -o mine.wav --reference me.wav      # clone a voice from a 5-15 s clip
"""
import argparse
import sys
from pathlib import Path

from .voices import ENGLISH


def main(argv=None):
    p = argparse.ArgumentParser(prog="dittodub-longform", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="text to read (or --file, or pipe it in)")
    p.add_argument("--file", help="read the text from this UTF-8 file")
    p.add_argument("-o", "--output", default="speech.wav", help="output WAV (default: speech.wav)")
    p.add_argument("--voice", default="US_Emma", help=f"built-in voice: {', '.join(ENGLISH)}, or a Chinese voice "
                   "such as 灵小柔 (default: US_Emma)")
    p.add_argument("--reference", help="clone the voice in this clip (5-15 s of one speaker, little noise)")
    p.add_argument("--reference-text", help="what is said in --reference (transcribed automatically if omitted)")
    p.add_argument("--language", help="en, zh, ja, ko, de, fr or es (default: zh if the text has Chinese, else en)")
    p.add_argument("--seed", type=int, default=11, help="another seed gives another reading of the same text")
    p.add_argument("--no-normalize", action="store_true", help="read the text as written (no spelled-out numbers)")
    p.add_argument("--model", default=None, help="local checkpoint directory or Hugging Face repo")
    p.add_argument("--no-fast", action="store_true", help="use stock MLX kernels only")
    p.add_argument("--list-voices", action="store_true", help="show the built-in voices and exit")
    a = p.parse_args(argv)
    if a.list_voices:
        print("English: " + ", ".join(ENGLISH))
        print("Chinese: 100+ Ming voices, for example 灵小柔 and 灵岩 (see Ming 3.0's voice catalog)")
        return 0
    text = Path(a.file).read_text(encoding="utf-8") if a.file else a.text
    if text is None and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text or not text.strip():
        p.error("give the text as an argument, with --file, or on stdin")
    if not a.reference and not (a.voice in ENGLISH or any("一" <= c <= "鿿" for c in a.voice)):
        p.error(f"unknown voice {a.voice!r}; try --list-voices")
    from . import MODEL, LongForm
    print("Loading Ming 3.0 TTS (the first run downloads about 2.8 GB)...", flush=True)
    tts = LongForm(a.model or MODEL, fast=not a.no_fast)
    print("Fast Metal kernels: " + ("on" if tts.fast else "off (stock MLX)"), flush=True)
    audio = tts.speak(text, a.voice, reference=a.reference, reference_text=a.reference_text, language=a.language,
                      seed=a.seed, normalize_text=not a.no_normalize, verbose=True)
    tts.save(audio, a.output)
    print(f"Saved {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
