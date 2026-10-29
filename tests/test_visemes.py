import time
from unittest.mock import MagicMock, patch

import numpy as np

from ovos_tts_plugin_audio8 import visemes
from ovos_tts_plugin_audio8.chunking import PlaybackClock
from ovos_tts_plugin_audio8.streaming import silence_pcm16, to_pcm16, wav_stream_header


RATE = 44100


def _tone(seconds: float, rms_db: float) -> np.ndarray:
    amplitude = 10 ** (rms_db / 20) * np.sqrt(2)
    return (amplitude * np.sin(2 * np.pi * 220 * np.arange(int(RATE * seconds)) / RATE)).astype(np.float32)


def test_loudness_ladder_maps_silence_to_rest_and_loud_to_wide_open():
    assert visemes.loudness_visemes(np.zeros(RATE // 10, dtype=np.float32), RATE) == [[4, 0.1]]
    assert visemes.loudness_visemes(_tone(0.08, -20), RATE)[0][0] == 0
    assert visemes.loudness_visemes(_tone(0.08, -34), RATE)[0][0] == 2
    assert visemes.loudness_visemes(_tone(0.08, -40), RATE)[0][0] == 3


def test_runs_merge_and_end_times_are_cumulative():
    audio = np.concatenate([np.zeros(int(RATE * 0.2), dtype=np.float32), _tone(0.12, -20)])
    assert visemes.loudness_visemes(audio, RATE) == [[4, 0.2], [0, 0.32]]


def test_visemes_survive_the_phonemes_slot():
    pairs = [[4, 0.2], [0, 0.32], [3, 0.4]]
    assert visemes.decode_visemes(visemes.encode_visemes(pairs)) == pairs
    assert visemes.decode_visemes(None) == []


def test_stream_callbacks_time_each_chunk_from_the_header_rate():
    bus = MagicMock()
    with patch("shutil.which", return_value="/opt/homebrew/bin/ffplay"):
        callbacks = visemes.VisemeStreamCallbacks(bus, play_args=["cat"])
    sent = []
    callbacks._schedule = lambda plays_at, pairs: sent.append((plays_at, pairs))
    with patch.object(visemes.StreamingTTSCallbacks, "stream_chunk"):
        callbacks.stream_chunk(wav_stream_header(RATE) + silence_pcm16(RATE, 0.2))
        callbacks.stream_chunk(to_pcm16(_tone(0.12, -20)))
    assert callbacks.sample_rate == RATE
    (first_at, first), (second_at, second) = sent
    assert first == [[4, 0.2]] and second == [[0, 0.12]]
    assert abs(second_at - first_at - 0.2) < 1e-6  # the second chunk starts after the first one's audio


def test_a_late_chunk_plays_on_arrival_not_on_the_stale_schedule():
    callbacks = visemes.VisemeStreamCallbacks(MagicMock(), play_args=["cat"])
    sent = []
    callbacks._schedule = lambda plays_at, pairs: sent.append(plays_at)
    times = iter([100.0, 101.0])  # the 0.2 s pad is written at t=100, the next chunk only arrives at t=101
    callbacks._clock = PlaybackClock(now=lambda: next(times))
    with patch.object(visemes.StreamingTTSCallbacks, "stream_chunk"):
        callbacks.stream_chunk(wav_stream_header(RATE) + silence_pcm16(RATE, 0.2))
        callbacks.stream_chunk(to_pcm16(_tone(0.12, -20)))
    latency = callbacks._clock.latency
    assert sent == [100.0 + latency, 101.0 + latency]


def test_scheduled_visemes_reach_the_bus():
    bus = MagicMock()
    callbacks = visemes.VisemeStreamCallbacks(bus, play_args=["cat"])
    callbacks._message = MagicMock()
    callbacks._schedule(time.monotonic(), [[0, 0.1]])
    callbacks._timers[0].join(1)
    name, data = callbacks._message.forward.call_args.args
    assert name == visemes.VISEME_MESSAGE and data["visemes"] == [[0, 0.1]]
    bus.emit.assert_called_once()
