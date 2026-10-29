"""How streamed audio reaches the speaker: the built-in player, or ffplay as a fallback.

Both send each chunk's visemes when it starts to play, and tell stream_tts how much silence to pad
the stream with and how late playback starts, so chunk scheduling uses the same timing model.
"""

import shutil
import struct
from typing import Callable, Optional, Tuple

from ovos_bus_client.message import Message, dig_for_message
from ovos_plugin_manager.templates.tts import StreamingTTSCallbacks
from ovos_utils.log import LOG

from ovos_tts_plugin_audio8.chunking import STREAM_PLAYBACK_LATENCY
from ovos_tts_plugin_audio8.player import PcmPlayer
from ovos_tts_plugin_audio8.streaming import STREAM_PAD_SECONDS
from ovos_tts_plugin_audio8.visemes import VisemeScheduler


STREAM_PLAYERS = ("auto", "sounddevice", "ffplay")
_WAV_HEADER_BYTES = 44
# Before the first chunk opens the stream; replaced by the device's reported latency (~20-30 ms measured).
_SOUNDDEVICE_LATENCY_GUESS = 0.03
# OPM's default `ffplay -autoexit -nodisp -` probes ~2 s of piped input before it starts playing
# (measured: first sound at 2.33 s vs 0.22 s with these flags, no audio dropped).
_LOW_LATENCY_FFPLAY_ARGS = ("-autoexit", "-nodisp", "-loglevel", "error",
                            "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer", "-")


def split_stream_chunk(chunk: bytes) -> Tuple[Optional[int], bytes]:
    """(sample rate if this chunk carries the WAV header, PCM payload)."""
    if chunk[:4] == b"RIFF":
        return struct.unpack_from("<I", chunk, 24)[0], chunk[_WAV_HEADER_BYTES:]
    return None, chunk


def ffplay_args(config: dict) -> Optional[list]:
    """Player command for streamed audio; None lets OPM pick paplay/aplay."""
    if config.get("stream_player_args"):
        return list(config["stream_player_args"])
    ffplay = shutil.which("ffplay")
    return [ffplay, *_LOW_LATENCY_FFPLAY_ARGS] if ffplay else None


class FfplayStreamCallbacks(StreamingTTSCallbacks):
    """OPM's subprocess player (ffplay/paplay/aplay on stdin), plus visemes. Kept as the fallback."""

    pad_seconds = STREAM_PAD_SECONDS  # ffplay drops ~0.1 s at the start and clips the end
    playback_latency = STREAM_PLAYBACK_LATENCY

    def __init__(self, bus, play_args=None, tts_config=None):
        super().__init__(bus, play_args=play_args, tts_config=tts_config)
        self.visemes = VisemeScheduler(bus)
        self.sample_rate: Optional[int] = None

    def stream_start(self, message=None):
        self.visemes.start(message or dig_for_message() or Message("speak"), self.playback_latency)
        super().stream_start(message)

    def stream_chunk(self, chunk):
        super().stream_chunk(chunk)
        rate, pcm = split_stream_chunk(chunk)
        self.sample_rate = rate or self.sample_rate
        if pcm and self.sample_rate:
            self.visemes.add(pcm, self.sample_rate)

    def stream_stop(self, listen=False, message=None):
        super().stream_stop(listen, message)  # waits for the player, so every chunk has played
        self.visemes.stop()


class SoundDeviceStreamCallbacks:
    """Plays the stream through PcmPlayer: silence on underrun, nothing written is ever dropped."""

    pad_seconds = 0.0

    def __init__(self, bus, tts_config=None, player_factory: Callable[[int], PcmPlayer] = PcmPlayer):
        self.bus = bus
        self.config = tts_config or {}
        self._player_factory = player_factory
        self.player: Optional[PcmPlayer] = None
        self.visemes = VisemeScheduler(bus)
        self._message = None

    @property
    def playback_latency(self) -> float:
        return self.player.latency if self.player else _SOUNDDEVICE_LATENCY_GUESS

    def stream_start(self, message=None):
        self._message = message or dig_for_message() or Message("speak")
        if not self.config.get("pulse_duck", False):
            self.bus.emit(self._message.forward("ovos.common_play.duck"))
        self.bus.emit(self._message.forward("recognizer_loop:audio_output_start"))
        self.visemes.start(self._message, self.playback_latency)

    def stream_chunk(self, chunk):
        rate, pcm = split_stream_chunk(chunk)
        if rate:
            self._open(rate)
        if pcm and self.player:
            self.player.write(pcm)
            self.visemes.add(pcm, self.player.sample_rate)

    def _open(self, sample_rate: int):
        if self.player:
            self.player.close()
        self.player = self._player_factory(sample_rate)
        self.visemes.clock.latency = self.player.latency

    def stream_stop(self, listen=False, message=None):
        message = message or self._message or Message("speak")
        if self.player:
            self.player.drain(timeout=30)
            if self.player.underruns:
                LOG.info("Audio8 stream ran dry %d time(s); played silence, dropped nothing", self.player.underruns)
            self.player = None
        self.visemes.stop()
        if not self.config.get("pulse_duck", False):
            self.bus.emit(message.forward("ovos.common_play.unduck"))
        self.bus.emit(message.forward("recognizer_loop:audio_output_end"))
        if listen:
            self.bus.emit(message.forward("mycroft.mic.listen"))


def make_stream_callbacks(bus, config: dict):
    """The built-in player when sounddevice and an output device exist, else ffplay (or `stream_player`)."""
    choice = config.get("stream_player", "auto")
    if choice not in STREAM_PLAYERS:
        raise ValueError(f"Unknown Audio8 stream_player '{choice}'. Choose one of {STREAM_PLAYERS}")
    if choice != "ffplay" and not (choice == "auto" and config.get("stream_player_args")):
        try:
            import sounddevice

            sounddevice.query_devices(kind="output")
            return SoundDeviceStreamCallbacks(bus, tts_config=config)
        except Exception as err:
            if choice == "sounddevice":
                raise
            LOG.info("Audio8 built-in stream player unavailable (%s); using ffplay", err)
    return FfplayStreamCallbacks(bus, play_args=ffplay_args(config), tts_config=config)
