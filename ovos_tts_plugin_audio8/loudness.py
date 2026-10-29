"""One output level for every cloned voice.

Audio8 copies the reference clip's loudness: across two clips 16.3 dB apart, the outputs were 16.1 dB
apart, each sitting 2.6-2.8 dB below its clip's speech level. So one gain per voice, computed from the
clip, brings every voice to the same level, and works for streaming (no adaptive gain to pump).
"""

from typing import Optional

import numpy as np


# Median RMS of 40 ms speech windows in the output. Peaks reached 25.7 dB above the median in the
# measured samples, so this leaves about 3 dB of headroom.
TARGET_SPEECH_DB = -29.0
OUTPUT_MINUS_REFERENCE_DB = -2.7
_WINDOW_SECONDS = 0.04
_SPEECH_FLOOR_DB = -50.0


def speech_level_db(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Median RMS (dBFS) of the 40 ms windows louder than -50 dB, or None if there are none."""
    hop = max(1, int(sample_rate * _WINDOW_SECONDS))
    windows = [audio[i: i + hop] for i in range(0, len(audio) - hop + 1, hop)]
    rms = np.array([np.sqrt(np.mean(np.square(w, dtype=np.float64))) for w in windows])
    speech = rms[rms > 10 ** (_SPEECH_FLOOR_DB / 20)]
    return float(20 * np.log10(np.median(speech))) if len(speech) else None


def voice_gain(reference_audio: np.ndarray, sample_rate: int) -> float:
    """Linear gain that brings this voice's output to TARGET_SPEECH_DB; 1.0 for a silent clip."""
    level = speech_level_db(reference_audio, sample_rate)
    if level is None:
        return 1.0
    return float(10 ** ((TARGET_SPEECH_DB - (level + OUTPUT_MINUS_REFERENCE_DB)) / 20))
