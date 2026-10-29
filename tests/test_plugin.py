import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovos_tts_plugin_audio8 import Audio8TTSPlugin, Audio8TTSPluginConfig, tts, visemes


VOICES = {
    "seven": {"reference_audio": "/voices/seven.wav", "reference_text": "Regeneration cycle complete."},
    "no_text": {"reference_audio": "/voices/x.wav"},
}


@pytest.fixture
def plugin():
    return Audio8TTSPlugin(config={"lang": "en-us", "voice": "seven", "voices": VOICES, "preload": False})


def _fake_engine(sample_rate=44100, seconds=1.0):
    engine = MagicMock()
    engine.sample_rate = sample_rate
    engine.device = "cpu"
    return engine, np.full(int(sample_rate * seconds), 0.5, dtype=np.float32)


def test_default_sentinel_falls_back_to_configured_voice(plugin):
    assert plugin._resolve_voice("default") == tts.VoiceProfile("/voices/seven.wav", "Regeneration cycle complete.")
    assert plugin._resolve_voice(None).is_cloned


def test_no_configured_voice_uses_unconditioned_model():
    assert not Audio8TTSPlugin(config={"lang": "en-us", "preload": False})._resolve_voice("default").is_cloned


def test_unknown_voice_fails_loudly(plugin):
    with pytest.raises(ValueError, match="Unknown Audio8 voice"):
        plugin._resolve_voice("picard")


def test_reference_audio_without_transcript_is_rejected(plugin):
    with pytest.raises(ValueError, match="reference_text"):
        plugin._resolve_voice("no_text")


def test_get_tts_writes_resampled_pcm16(plugin, tmp_path):
    engine, audio = _fake_engine()
    out = tmp_path / "out.wav"
    with patch.object(tts, "_get_engine", return_value=engine), patch.object(tts, "_synthesize", return_value=audio):
        result = plugin.get_tts("Engage.", str(out))
    assert result[0] == str(out)
    assert visemes.decode_visemes(result[1])[-1] == [0, 1.0]  # constant 0.5 amplitude: wide open for 1 s
    with wave.open(str(out)) as wf:
        assert (wf.getsampwidth(), wf.getnchannels(), wf.getframerate()) == (2, 1, tts.DEFAULT_SAMPLE_RATE)
        assert wf.getnframes() == tts.DEFAULT_SAMPLE_RATE


def test_cloned_voice_passes_codes_through_processor():
    import torch

    engine = MagicMock()
    engine.device = "cpu"
    engine.processor.return_value = {}
    engine.model.decode_audio.return_value = (torch.zeros(1, 10), torch.tensor([10]))
    engine.model.generate.return_value.codes = torch.zeros((1, 10, 5), dtype=torch.long)
    voice = tts.VoiceProfile("/voices/seven.wav", "Regeneration cycle complete.")
    with patch.object(tts, "_encode_reference", return_value="CODES"):
        audio = tts._synthesize(engine, "Engage.", voice, {}, decoder="model")
    assert audio.shape == (10,)
    assert engine.model.decode_audio.call_args.args[0].shape[-1] == 4  # junk final frame dropped
    kwargs = engine.processor.call_args.kwargs
    assert kwargs["reference_codes"] == ["CODES"]
    assert kwargs["reference_text"] == ["Regeneration cycle complete."]


GEN_KWARGS = {"max_new_tokens": 512, "temperature": 0.8, "top_p": 0.95, "top_k": 50}


@pytest.mark.parametrize("decoder", ["cuda_graph", "compile", "eager"])
def test_frame_decoders_route_through_engine(decoder):
    engine = MagicMock()
    codes = tts._generate_codes(engine, {"x": 1}, GEN_KWARGS, decoder)
    engine.frame_decoder.assert_called_once_with(decoder, top_k=50, top_p=0.95, temperature=0.8)
    engine.frame_decoder.return_value.generate.assert_called_once_with({"x": 1}, 512)
    assert codes is engine.frame_decoder.return_value.generate.return_value
    engine.model.generate.assert_not_called()


def test_model_decoder_invalidates_frame_decoder():
    engine = MagicMock()
    engine.decoder = "stale"
    tts._generate_codes(engine, {}, GEN_KWARGS, "model")
    assert engine.decoder is None
    engine.model.generate.assert_called_once()


@pytest.mark.parametrize(("device", "expected"), [("cuda", "cuda_graph"), ("mps", "compile"), ("cpu", "compile")])
def test_auto_decoder_picks_fastest_measured(device, expected):
    assert tts._pick_decoder("auto", device) == expected


def test_explicit_decoder_wins_over_auto():
    assert tts._pick_decoder("eager", "mps") == "eager"


def _engine_with_fake_decoder(side_effect):
    from ovos_tts_plugin_audio8 import frame_decoder

    engine = tts.Audio8Engine(model=MagicMock(), processor=MagicMock(), device="mps")
    built = []

    def fake(model, top_k, top_p, temperature, mode):
        built.append(mode)
        if side_effect and mode is frame_decoder.DecodeMode.COMPILE:
            raise side_effect
        decoder = MagicMock()
        decoder.mode, decoder.sampling = mode, (top_k, top_p, temperature)
        return decoder

    return engine, built, patch.object(frame_decoder, "FrameDecoder", side_effect=fake)


def test_compile_failure_falls_back_to_eager():
    from ovos_tts_plugin_audio8.frame_decoder import DecodeMode

    engine, built, patched = _engine_with_fake_decoder(RuntimeError("inductor codegen"))
    with patched:
        decoder = engine.frame_decoder("compile", 50, 0.95, 0.8)
        again = engine.frame_decoder("compile", 50, 0.95, 0.8)
    assert built == [DecodeMode.COMPILE, DecodeMode.EAGER]
    assert decoder.mode is DecodeMode.EAGER and again is decoder


def test_non_compile_failures_propagate():
    engine, _, patched = _engine_with_fake_decoder(None)
    with patched, patch("ovos_tts_plugin_audio8.frame_decoder.FrameDecoder", side_effect=RuntimeError("cuda")):
        with pytest.raises(RuntimeError, match="cuda"):
            engine.frame_decoder("cuda_graph", 50, 0.95, 0.8)


def test_unknown_decoder_fails_at_init():
    with pytest.raises(ValueError, match="Unknown Audio8 decoder"):
        Audio8TTSPlugin(config={"lang": "en-us", "decoder": "warp"})


def test_preload_synthesizes_once_at_init():
    engine, audio = _fake_engine()
    with patch.object(tts, "_get_engine", return_value=engine), \
            patch.object(tts, "_synthesize", return_value=audio) as synth:
        Audio8TTSPlugin(config={"lang": "en-us", "voice": "seven", "voices": VOICES, "preload": True})
    synth.assert_called_once()
    assert synth.call_args.args[2].is_cloned


def test_config_advertises_english():
    assert Audio8TTSPlugin.available_languages() == {"en"}
    assert Audio8TTSPluginConfig["en"][0]["meta"]["offline"] is True
