"""Render Star Trek sample phrases through the plugin with the bundled cloned voice.

Usage: uv run python scripts/generate_samples.py [--device auto|cpu|mps|cuda] [--decoder auto|...] [--out samples]
"""

import argparse
import time
import wave
from pathlib import Path

from ovos_tts_plugin_audio8 import Audio8TTSPlugin


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_AUDIO = REPO_ROOT / "voices" / "regeneration_cycle.wav"
REFERENCE_TEXT = "Oh six hundred hours, regeneration cycle complete."

PHRASES = [
    "Captain, long range sensors have detected a Borg cube on an intercept course.",
    "Resistance is futile. Your biological and technological distinctiveness will be added to our own.",
    "Shields are at forty percent. I recommend we withdraw to the nebula.",
    "The warp core breach is imminent. All hands, abandon ship.",
    "I have analyzed the anomaly. It appears to be a rupture in subspace.",
    "Fascinating. The humans persist in their irrational optimism.",
    "Engage maximum warp. Set a course for the Alpha Quadrant.",
    "Computer, begin auto destruct sequence. Authorization Janeway, pi one one zero.",
]


def _duration(path: Path) -> float:
    with wave.open(str(path)) as wf:
        return wf.getnframes() / wf.getframerate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(REPO_ROOT / "samples"))
    parser.add_argument("--decoder", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plugin = Audio8TTSPlugin(config={
        "lang": "en-us",
        "device": args.device,
        "decoder": args.decoder,
        "voice": "seven",
        "voices": {"seven": {"reference_audio": str(REFERENCE_AUDIO), "reference_text": REFERENCE_TEXT}},
    })
    for index, phrase in enumerate(PHRASES, start=1):
        wav_path = out_dir / f"{index:02d}.wav"
        started = time.time()
        plugin.get_tts(phrase, str(wav_path))
        elapsed = time.time() - started
        audio_seconds = _duration(wav_path)
        rtf = elapsed / audio_seconds
        print(f"{wav_path.name}  {audio_seconds:5.2f}s audio  {elapsed:5.1f}s wall  RTF {rtf:.2f}  {phrase}")


if __name__ == "__main__":
    main()
