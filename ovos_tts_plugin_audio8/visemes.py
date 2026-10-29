"""Loudness-driven mouth movement for Audio8, which has no phoneme alignment of its own.

Each 40 ms window's RMS picks a Mycroft viseme code on an openness ladder: closed (4), half
open (3), rounded (2), wide (0). Steps sit at percentiles of measured Audio8 speech (about the
70th, 40th and 15th), expressed relative to the normalized speech level in loudness.py.
"""

import threading
import time
from typing import List, Optional

import numpy as np

from ovos_tts_plugin_audio8.chunking import PlaybackClock
from ovos_tts_plugin_audio8.loudness import TARGET_SPEECH_DB


WINDOW_SECONDS = 0.04
REST_CODE = 4
# (minimum RMS dB relative to the speech median, code), loudest first. Measured on speech with a
# -35.6 dB median: wide from -31, round from -37, half open from -44.
_LADDER = ((TARGET_SPEECH_DB + 4.6, 0), (TARGET_SPEECH_DB - 1.4, 2), (TARGET_SPEECH_DB - 8.4, 3))
VISEME_MESSAGE = "enclosure.mouth.viseme_list"


def _code_for(rms_db: float) -> int:
    return next((code for floor, code in _LADDER if rms_db >= floor), REST_CODE)


def loudness_visemes(audio: np.ndarray, sample_rate: int) -> List[List[float]]:
    """[[code, cumulative end seconds], ...] for float audio, merging runs of the same code."""
    hop = max(1, int(sample_rate * WINDOW_SECONDS))
    pairs: List[List[float]] = []
    for start in range(0, len(audio), hop):
        window = audio[start: start + hop]
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        code = _code_for(20 * np.log10(max(rms, 1e-9)))
        end = round(min(start + hop, len(audio)) / sample_rate, 3)
        if pairs and pairs[-1][0] == code:
            pairs[-1][1] = end
        else:
            pairs.append([code, end])
    return pairs


def encode_visemes(pairs: List[List[float]]) -> str:
    """Carry visemes through OVOS's `phonemes` slot (and its cache) as "code:end" pairs."""
    return " ".join(f"{int(code)}:{end}" for code, end in pairs)


def decode_visemes(encoded: Optional[str]) -> List[List[float]]:
    pairs = []
    for item in (encoded or "").split():
        code, end = item.split(":")
        pairs.append([int(code), float(end)])
    return pairs


class VisemeScheduler:
    """Sends each chunk's visemes when that chunk starts to play, timed by a playback clock."""

    def __init__(self, bus):
        self.bus = bus
        self.clock = PlaybackClock()
        self._timers: List[threading.Timer] = []
        self._message = None

    def start(self, message, latency: float):
        self.stop()
        self.clock = PlaybackClock(latency=latency, now=self.clock.now)
        self._message = message

    def add(self, pcm: bytes, sample_rate: int):
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        plays_at = self.clock.add(len(audio) / sample_rate)
        self._schedule(plays_at, loudness_visemes(audio, sample_rate))

    def stop(self):
        for timer in self._timers:
            timer.cancel()
        self._timers = []

    def _schedule(self, plays_at: float, pairs):
        timer = threading.Timer(max(0.0, plays_at - time.monotonic()), self._emit, args=(pairs,))
        timer.daemon = True
        self._timers.append(timer)
        timer.start()

    def _emit(self, pairs):
        self.bus.emit(self._message.forward(VISEME_MESSAGE, {"start": time.time(), "visemes": pairs}))
