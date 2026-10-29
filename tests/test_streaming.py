import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ovos_tts_plugin_audio8 import Audio8TTSPlugin, streaming, tts
from ovos_tts_plugin_audio8.chunking import CostModel, PlaybackClock, sustainable_chunk


FRAME_SAMPLES = 4


class FakeCausalCodecModel:
    """decode_audio where frame t depends on every earlier frame, like the real codec's receptive field.

    Any chunking that decodes a window not starting at frame 0 produces wrong samples here.
    """

    config = SimpleNamespace(codec_frame_size=FRAME_SAMPLES, codec_sample_rate=86)  # ~46 ms frames, like Audio8

    def decode_audio(self, codes):
        values = torch.cumsum(codes[0, 0].double(), dim=0)
        waveform = values.repeat_interleave(FRAME_SAMPLES).float()[None]
        return waveform, torch.tensor([waveform.shape[1]])


class FakeDecoder:
    def __init__(self, total_frames: int):
        self.codes_out = torch.randint(0, 4096, (10, 512))
        self.total = total_frames

    def iter_generate(self, inputs, max_new_tokens):
        yield from range(8, self.total, 8)
        yield self.total


def _clock(queued_seconds: float) -> PlaybackClock:
    """A frozen clock with `queued_seconds` of audio waiting in the player, mid-utterance."""
    clock = PlaybackClock(latency=0.0, now=lambda: 0.0)
    clock.playhead = queued_seconds
    clock.speech_written = True
    return clock


class FixedCost(CostModel):
    """MPS-like costs that the instant fakes cannot overwrite."""

    def __init__(self, decode=0.005):
        super().__init__()
        self.generate, self.decode = 0.025, decode

    def observe_generation(self, frames, seconds):
        pass

    def observe_decode(self, frames, seconds):
        pass


def _chunk_frames(total_frames: int, queued_seconds: float, cost=None):
    model, decoder = FakeCausalCodecModel(), FakeDecoder(total_frames)
    chunks = streaming.stream_codes_to_audio(model, decoder, {}, 1024, _clock(queued_seconds), cost or FixedCost())
    return [len(chunk) // FRAME_SAMPLES for chunk in chunks], model, decoder


@pytest.mark.parametrize("queued_seconds", [0.0, 100.0])
@pytest.mark.parametrize("total_frames", [5, 12, 100, 301])
def test_streamed_chunks_equal_full_decode(total_frames, queued_seconds):
    model, decoder = FakeCausalCodecModel(), FakeDecoder(total_frames)
    chunks = streaming.stream_codes_to_audio(model, decoder, {}, 1024, _clock(queued_seconds), CostModel())
    playable = total_frames - streaming.TRAILING_JUNK_FRAMES
    full = model.decode_audio(decoder.codes_out[:, :playable].unsqueeze(0))[0][0].numpy()
    np.testing.assert_array_equal(np.concatenate(list(chunks)), full)


def test_a_starving_player_gets_chunks_that_grow_with_the_prefix():
    sizes, _, _ = _chunk_frames(300, queued_seconds=0.0)
    assert sizes[0] == 8 - streaming.TRAILING_JUNK_FRAMES  # the very first check, holding back the newest frame
    emitted = sizes[0]
    for size in sizes[1:-1]:
        assert size >= sustainable_chunk(emitted + size, 4 / 86, FixedCost())
        emitted += size
    assert sizes[-2] > sizes[1]
    assert sum(sizes) == 300 - streaming.TRAILING_JUNK_FRAMES


def test_cheap_decoding_streams_at_every_check():
    sizes, _, _ = _chunk_frames(100, queued_seconds=0.0, cost=FixedCost(decode=0.0))
    assert all(size == 8 for size in sizes[1:-1])


def test_a_well_fed_player_waits_for_one_final_decode():
    sizes, _, _ = _chunk_frames(100, queued_seconds=100.0)
    assert sizes == [100 - streaming.TRAILING_JUNK_FRAMES]


def test_finalized_stream_header_is_a_valid_wav(tmp_path):
    path = tmp_path / "stream.wav"
    pcm = streaming.to_pcm16(np.linspace(-1, 1, 1000, dtype=np.float32))
    path.write_bytes(streaming.wav_stream_header(44100) + pcm)
    streaming.finalize_wav_header(str(path))
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (44100, 1, 2)
        assert wav.getnframes() == 1000


def test_finalize_rejects_truncated_file(tmp_path):
    path = tmp_path / "short.wav"
    path.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="too short"):
        streaming.finalize_wav_header(str(path))


def test_get_tts_synthesizes_each_sentence(tmp_path):
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False})
    engine = SimpleNamespace(sample_rate=44100, device="cpu")
    calls = []

    def fake_synthesize(engine, sentence, voice, gen_kwargs, decoder):
        calls.append(sentence)
        return np.zeros(441, dtype=np.float32)

    with patch.object(tts, "_get_engine", return_value=engine), patch.object(tts, "_synthesize", fake_synthesize):
        plugin.get_tts("Shields are at forty percent. Engage maximum warp.", str(tmp_path / "out.wav"))
    assert calls == ["Shields are at forty percent.", "Engage maximum warp."]


def _init_streaming(config):
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False, "enable_streaming": True, **config})
    with patch("shutil.which", return_value="/opt/homebrew/bin/ffplay"), patch.object(tts.TTS, "playback", None):
        plugin.init(bus=MagicMock(), playback=MagicMock())
    return plugin


def test_streaming_init_uses_low_latency_ffplay():
    args = _init_streaming({}).callbacks.play_args
    assert args[0] == "/opt/homebrew/bin/ffplay"
    assert args[args.index("-probesize") + 1] == "32" and args[-1] == "-"


def test_stream_player_args_override():
    assert _init_streaming({"stream_player_args": ["paplay", "-"]}).callbacks.play_args == ["paplay", "-"]


def test_init_without_streaming_does_not_need_a_player():
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False})
    # TTS.init stores the PlaybackThread on the class; patch it so the mock doesn't leak.
    with patch("shutil.which", return_value=None), patch.object(tts.TTS, "playback", None):
        plugin.init(bus=None, playback=MagicMock())
    assert not hasattr(plugin, "callbacks")
