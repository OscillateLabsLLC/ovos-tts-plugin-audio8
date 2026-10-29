import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovos_tts_plugin_audio8 import Audio8TTSPlugin, stream_players, tts
from ovos_tts_plugin_audio8.player import BLOCK_FRAMES, SAMPLE_BYTES, PcmPlayer
from ovos_tts_plugin_audio8.streaming import silence_pcm16, wav_stream_header


RATE = 44100
BLOCK_BYTES = BLOCK_FRAMES * SAMPLE_BYTES


class FakeStream:
    latency = 0.02

    def __init__(self, sample_rate, callback):
        self.sample_rate, self.callback = sample_rate, callback
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def pull(self) -> bytes:
        """One device callback, as CoreAudio would make it."""
        out = bytearray(BLOCK_BYTES)
        self.callback(out, BLOCK_FRAMES, None, None)
        return bytes(out)


def _player():
    streams = []

    def factory(rate, callback):
        streams.append(FakeStream(rate, callback))
        return streams[-1]

    return PcmPlayer(RATE, stream_factory=factory), streams


def test_player_plays_everything_written_in_order():
    player, (stream,) = _player()
    pcm = bytes(range(256)) * (BLOCK_BYTES // 128)  # two blocks
    player.write(pcm[:300])
    player.write(pcm[300:])
    assert stream.started and player.latency == FakeStream.latency
    assert stream.pull() + stream.pull() == pcm
    assert player.underruns == 0


def test_running_dry_plays_silence_and_keeps_the_rest():
    player, (stream,) = _player()
    stream.pull()  # before any audio: startup, not an underrun
    player.write(b"\x01" * 100)
    block = stream.pull()
    assert block[:100] == b"\x01" * 100 and block[100:] == bytes(BLOCK_BYTES - 100)
    stream.pull()  # still dry: the same underrun, not a second one
    player.write(b"\x02" * BLOCK_BYTES)
    assert stream.pull() == b"\x02" * BLOCK_BYTES
    assert player.underruns == 1


def test_running_out_while_draining_is_not_an_underrun():
    player, (stream,) = _player()
    player.write(b"\x01" * 100)
    with patch("ovos_tts_plugin_audio8.player.time.sleep", lambda _: stream.pull()):
        player.drain(timeout=1)
    assert player.underruns == 0 and player.buffered_seconds() == 0 and stream.closed


class FakePlayer:
    latency = 0.025

    def __init__(self, sample_rate):
        self.sample_rate, self.written, self.underruns = sample_rate, [], 0
        self.drained = False

    def write(self, pcm):
        self.written.append(pcm)

    def drain(self, timeout=None):
        self.drained = True

    def close(self):
        pass


def _bus_names(bus):
    return [call.args[0].msg_type for call in bus.emit.call_args_list]


def test_sounddevice_callbacks_feed_the_player_and_signal_the_bus():
    bus = MagicMock()
    callbacks = stream_players.SoundDeviceStreamCallbacks(bus, player_factory=FakePlayer)
    callbacks.visemes = MagicMock()
    callbacks.stream_start()
    pcm = silence_pcm16(RATE, 0.1)
    callbacks.stream_chunk(wav_stream_header(RATE) + pcm)
    player = callbacks.player
    assert callbacks.playback_latency == FakePlayer.latency and player.written == [pcm]
    callbacks.stream_stop(listen=True)
    assert player.drained
    assert _bus_names(bus) == ["ovos.common_play.duck", "recognizer_loop:audio_output_start",
                               "ovos.common_play.unduck", "recognizer_loop:audio_output_end", "mycroft.mic.listen"]


def _fake_sounddevice(has_output=True):
    def query_devices(kind=None):
        if not has_output:
            raise ValueError("no output device")
        return {"name": "Speakers"}

    return SimpleNamespace(query_devices=query_devices)


@pytest.mark.parametrize("config, has_output, expected", [
    ({}, True, stream_players.SoundDeviceStreamCallbacks),
    ({}, False, stream_players.FfplayStreamCallbacks),
    ({"stream_player": "ffplay"}, True, stream_players.FfplayStreamCallbacks),
    ({"stream_player_args": ["paplay", "-"]}, True, stream_players.FfplayStreamCallbacks),
])
def test_factory_picks_the_built_in_player_when_it_can(config, has_output, expected):
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(has_output)}):
        assert type(stream_players.make_stream_callbacks(MagicMock(), config)) is expected


def test_asking_for_sounddevice_without_a_device_fails_loudly():
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(has_output=False)}):
        with pytest.raises(ValueError, match="no output device"):
            stream_players.make_stream_callbacks(MagicMock(), {"stream_player": "sounddevice"})


def test_unknown_stream_player_is_rejected():
    with pytest.raises(ValueError, match="Unknown Audio8 stream_player"):
        stream_players.make_stream_callbacks(MagicMock(), {"stream_player": "vlc"})


def test_ffplay_callbacks_schedule_visemes_from_the_header_rate():
    with patch("shutil.which", return_value="/opt/homebrew/bin/ffplay"):
        callbacks = stream_players.FfplayStreamCallbacks(MagicMock(), play_args=["cat"])
    callbacks.visemes = MagicMock()
    pcm = silence_pcm16(RATE, 0.2)
    with patch.object(stream_players.StreamingTTSCallbacks, "stream_chunk"):
        callbacks.stream_chunk(wav_stream_header(RATE) + pcm)
    callbacks.visemes.add.assert_called_once_with(pcm, RATE)


def _streamed_bytes(callbacks) -> list:
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False})
    plugin.callbacks = callbacks
    engine = SimpleNamespace(sample_rate=RATE, device="cpu")
    speech = np.full(441, 0.5, dtype=np.float32)

    async def collect():
        return [chunk async for chunk in plugin.stream_tts("Engage.")]

    with patch.object(tts.Audio8TTSPlugin, "_engine", engine, create=True), \
            patch.object(plugin, "_resolve_voice"), patch.object(plugin, "_decoder_for"), \
            patch.object(tts, "_stream_sentence", lambda *args: iter([speech])):
        return asyncio.run(collect())


def test_stream_pad_follows_the_player():
    header_bytes = len(wav_stream_header(RATE))
    built_in = _streamed_bytes(SimpleNamespace(pad_seconds=0.0, playback_latency=0.03))
    assert len(built_in) == 2 and len(built_in[0]) == header_bytes
    ffplay = _streamed_bytes(SimpleNamespace(pad_seconds=0.2, playback_latency=0.1))
    assert len(ffplay[0]) - header_bytes == len(ffplay[-1]) == len(silence_pcm16(RATE, 0.2))
