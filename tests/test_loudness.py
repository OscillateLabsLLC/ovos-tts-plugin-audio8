import numpy as np

from ovos_tts_plugin_audio8 import loudness


RATE = 44100


def _tone(rms_db: float, seconds: float = 1.0) -> np.ndarray:
    amplitude = 10 ** (rms_db / 20) * np.sqrt(2)
    return (amplitude * np.sin(2 * np.pi * 220 * np.arange(int(RATE * seconds)) / RATE)).astype(np.float32)


def test_speech_level_ignores_silence():
    audio = np.concatenate([np.zeros(RATE, dtype=np.float32), _tone(-30.0)])
    assert abs(loudness.speech_level_db(audio, RATE) - -30.0) < 0.1
    assert loudness.speech_level_db(np.zeros(RATE, dtype=np.float32), RATE) is None


def test_loud_and_quiet_clips_land_on_the_same_output_level():
    for clip_db in (-34.2, -17.9):  # the two measured reference clips
        gain_db = 20 * np.log10(loudness.voice_gain(_tone(clip_db), RATE))
        expected_output = clip_db + loudness.OUTPUT_MINUS_REFERENCE_DB + gain_db
        assert abs(expected_output - loudness.TARGET_SPEECH_DB) < 0.1


def test_silent_clip_leaves_gain_alone():
    assert loudness.voice_gain(np.zeros(RATE, dtype=np.float32), RATE) == 1.0
