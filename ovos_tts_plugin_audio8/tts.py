"""Audio8 TTS plugin for OVOS — 0.6B DualAR TTS with one-shot voice cloning."""

import os
import shutil
import wave
from dataclasses import dataclass
from typing import AsyncIterable, Dict, Iterator, Optional, Tuple

import numpy as np
from ovos_plugin_manager.templates.tts import TTS, StreamingTTS
from ovos_utils.log import LOG

from ovos_tts_plugin_audio8.chunking import CostModel, PlaybackClock
from ovos_tts_plugin_audio8.loudness import voice_gain
from ovos_tts_plugin_audio8.prewarm import Prewarmer
from ovos_tts_plugin_audio8.streaming import (
    STREAM_PAD_SECONDS,
    finalize_wav_header,
    playable_frames,
    silence_pcm16,
    stream_codes_to_audio,
    to_pcm16,
    wav_stream_header,
)
from ovos_tts_plugin_audio8.text import split_for_streaming, split_sentences
from ovos_tts_plugin_audio8.visemes import VisemeStreamCallbacks, decode_visemes, encode_visemes, loudness_visemes


DEFAULT_MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
# Pinned because the model ships custom code loaded via trust_remote_code;
# bump only after reviewing the upstream *.py files at the new commit.
DEFAULT_REVISION = "1e857568bfd4f02931aca3988ee773a44d2d26d3"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_MAX_NEW_TOKENS = 1024
_WARMUP_SENTENCE = "Ready."

# Engine-level caches shared across plugin instances. Keyed by
# (model_id, revision, device) and (engine id, reference_audio path).
_engines: Dict[tuple, "Audio8Engine"] = {}
_reference_codes: Dict[tuple, "object"] = {}
_voice_gains: Dict[tuple, float] = {}


@dataclass(frozen=True)
class VoiceProfile:
    """A cloned voice: a reference recording plus its exact transcript."""

    reference_audio: Optional[str]
    reference_text: Optional[str]

    @property
    def is_cloned(self) -> bool:
        return bool(self.reference_audio)


@dataclass
class Audio8Engine:
    model: object
    processor: object
    device: str
    # Decoders own the model's KV caches, so only the most recently built one is valid.
    decoder: Optional[object] = None

    @property
    def sample_rate(self) -> int:
        return int(self.model.config.codec_sample_rate)

    def frame_decoder(self, mode: str, top_k: int, top_p: float, temperature: float):
        from ovos_tts_plugin_audio8.frame_decoder import DecodeMode, FrameDecoder

        mode = DecodeMode(mode)
        sampling = (top_k, top_p, temperature)
        current = self.decoder
        if current is not None and current.requested_mode is mode and current.sampling == sampling:
            return current
        LOG.info("Building Audio8 %s decoder (top_k=%s, top_p=%s, temperature=%s)", mode.value, *sampling)
        try:
            self.decoder = FrameDecoder(self.model, *sampling, mode)
        except Exception:
            if mode is not DecodeMode.COMPILE:
                raise
            # Inductor's MPS backend is young; a codegen failure must not take the voice down.
            LOG.exception("Audio8 torch.compile failed; falling back to the eager decoder (~25% slower)")
            self.decoder = FrameDecoder(self.model, *sampling, DecodeMode.EAGER)
        self.decoder.requested_mode = mode
        return self.decoder


DECODER_CHOICES = ("auto", "cuda_graph", "compile", "eager", "model")


def _pick_decoder(requested: str, device: str) -> str:
    """Fastest measured decoder per device: CUDA graph on CUDA (RTF 0.27 on a 3090), torch.compile
    elsewhere (M3 Max MPS 0.54, CPU 1.49). Compile costs ~45s once; ``preload`` hides it."""
    if requested not in DECODER_CHOICES:
        raise ValueError(f"Unknown Audio8 decoder '{requested}'. Choose one of {DECODER_CHOICES}")
    if requested != "auto":
        return requested
    return "cuda_graph" if device == "cuda" else "compile"


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # Measured on M3 Max: MPS beats CPU here (0.68 vs 1.98 RTF), unlike Kokoro where MPS loses.
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_engine(model_id: str, revision: str, device: str) -> Audio8Engine:
    key = (model_id, revision, device)
    if key in _engines:
        return _engines[key]
    import torch
    from transformers import AutoModel, AutoProcessor

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        LOG.warning("Audio8 TTS on CPU is slower than real time (RTF ~1.5 on an M3 Max). Use CUDA or Apple MPS.")
    LOG.info("Loading Audio8 TTS %s@%s on %s (%s)", model_id, revision[:8], device, dtype)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True, dtype=dtype)
    restore_rope_buffers(model)
    _engines[key] = Audio8Engine(model=model.eval().to(device), processor=processor, device=device)
    return _engines[key]


def restore_rope_buffers(model):
    """Recompute Audio8's RoPE tables after loading.

    transformers 5 builds models on the meta device, so non-persistent buffers computed in
    ``__init__`` are never filled: ``freqs_cis`` comes out as garbage and ``fast_freqs_cis`` as zeros,
    and the model then never emits EOS. On transformers 4 this is a no-op rewrite of equal values.
    """
    import sys

    import torch

    precompute = sys.modules[type(model).__module__]._precompute_rope
    cfg = model.config
    with torch.no_grad():
        model.freqs_cis.copy_(precompute(cfg.max_seq_len, cfg.head_dim, cfg.rope_base))
        model.fast_freqs_cis.copy_(precompute(cfg.num_codebooks, cfg.fast_head_dim, cfg.rope_base))


def _encode_reference(engine: Audio8Engine, voice: VoiceProfile):
    """Encode the reference clip through the codec once; returns [num_codebooks, T] codes."""
    import torch

    key = (id(engine), voice.reference_audio)
    if key not in _reference_codes:
        LOG.info("Encoding Audio8 reference voice: %s", voice.reference_audio)
        audio = engine.processor._load_audio(voice.reference_audio, None)
        lengths = torch.tensor([audio.numel()], device=engine.device)
        codes, code_lengths = engine.model.encode_audio(audio.view(1, 1, -1).to(engine.device), lengths)
        _reference_codes[key] = codes[0, :, : int(code_lengths[0])].cpu()
        _voice_gains[key] = voice_gain(audio.numpy(), engine.processor.audio_sampling_rate)
        LOG.info("Audio8 voice %s: output gain %+.1f dB", voice.reference_audio,
                 20 * np.log10(_voice_gains[key]))
    return _reference_codes[key]


def _voice_gain_for(engine: Audio8Engine, voice: VoiceProfile) -> float:
    """Gain that brings this voice to the shared output level; the model's own voice is left as is."""
    if not voice.is_cloned:
        return 1.0
    _encode_reference(engine, voice)
    return _voice_gains[(id(engine), voice.reference_audio)]


def _build_inputs(engine: Audio8Engine, sentence: str, voice: VoiceProfile) -> dict:
    if voice.is_cloned:
        # Must go through the processor: it picks the cloning prompt template
        # only when reference_codes/audio is passed to it.
        inputs = engine.processor(
            text=[sentence],
            reference_text=[voice.reference_text],
            reference_codes=[_encode_reference(engine, voice)],
            return_tensors="pt",
        )
    else:
        inputs = engine.processor(text=[sentence], return_tensors="pt")
    return {name: value.to(engine.device) for name, value in inputs.items()}


def _generate_codes(engine: Audio8Engine, inputs: dict, gen_kwargs: dict, decoder: str):
    if decoder != "model":
        sampling = {key: gen_kwargs[key] for key in ("top_k", "top_p", "temperature")}
        return engine.frame_decoder(decoder, **sampling).generate(inputs, gen_kwargs["max_new_tokens"])
    # Upstream generate() replaces the KV caches a frame decoder (and its CUDA graph) points at.
    engine.decoder = None
    output = engine.model.generate(**inputs, do_sample=True, return_dict_in_generate=True, **gen_kwargs)
    return output.codes


def _synthesize(engine: Audio8Engine, sentence: str, voice: VoiceProfile, gen_kwargs: dict,
                decoder: str) -> np.ndarray:
    import torch

    inputs = _build_inputs(engine, sentence, voice)
    with torch.inference_mode():
        codes = _generate_codes(engine, inputs, gen_kwargs, decoder)
        waveforms, lengths = engine.model.decode_audio(codes[..., : playable_frames(codes.shape[-1])])
    return waveforms[0, : int(lengths[0])].float().cpu().numpy() * _voice_gain_for(engine, voice)


# OPM's default `ffplay -autoexit -nodisp -` probes ~2 s of piped input before it starts playing
# (measured: first sound at 2.33 s vs 0.22 s with these flags, no audio dropped).
_LOW_LATENCY_FFPLAY_ARGS = ("-autoexit", "-nodisp", "-loglevel", "error",
                            "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer", "-")


def _stream_player_args(config: dict) -> Optional[list]:
    """Player command for streamed audio; None lets OPM pick paplay/aplay."""
    if config.get("stream_player_args"):
        return list(config["stream_player_args"])
    ffplay = shutil.which("ffplay")
    return [ffplay, *_LOW_LATENCY_FFPLAY_ARGS] if ffplay else None



def _stream_sentence(engine: Audio8Engine, sentence: str, voice: VoiceProfile, gen_kwargs: dict,
                     decoder: str, clock: PlaybackClock, cost: CostModel) -> Iterator[np.ndarray]:
    """Yield codec-rate audio chunks for one sentence as frames are generated."""
    if decoder == "model":
        # Upstream generate() has no incremental hook; stream the sentence as one chunk.
        yield _synthesize(engine, sentence, voice, gen_kwargs, decoder)
        return
    inputs = _build_inputs(engine, sentence, voice)
    sampling = {key: gen_kwargs[key] for key in ("top_k", "top_p", "temperature")}
    frame_decoder = engine.frame_decoder(decoder, **sampling)
    gain = _voice_gain_for(engine, voice)
    for chunk in stream_codes_to_audio(engine.model, frame_decoder, inputs, gen_kwargs["max_new_tokens"], clock, cost):
        yield chunk * gain


def _resample(audio: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    if native_rate == target_rate:
        return audio
    from scipy.signal import resample_poly

    divisor = np.gcd(native_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, native_rate // divisor).astype(np.float32)


def _write_pcm16_wav(path: str, audio: np.ndarray, sample_rate: int):
    # int16 PCM, not IEEE float: many OVOS playback paths can't handle format tag 3.
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class Audio8TTSPlugin(StreamingTTS):
    """Audio8 TTS — DualAR 0.6B TTS with one-shot voice cloning.

    Config example (mycroft.conf):
        "tts": {
            "module": "ovos-tts-plugin-audio8",
            "ovos-tts-plugin-audio8": {
                "voice": "seven",
                "voices": {
                    "seven": {
                        "reference_audio": "/path/to/clip.wav",
                        "reference_text": "Exact transcript of the clip."
                    }
                }
            }
        }

    Voices without a reference_audio use the model's unconditioned default voice.
    """

    def __init__(self, config=None):
        super().__init__(config=config, audio_ext="wav")
        if self.config.get("decoder", "auto") not in DECODER_CHOICES:
            raise ValueError(f"Unknown Audio8 decoder '{self.config['decoder']}'. Choose one of {DECODER_CHOICES}")
        # On by default: without it the first sentence pays model load + compile (10-60s).
        if self.config.get("preload", True):
            self._warm_up()

    def _warm_up(self):
        """Pay model load, reference encoding, codec load and decoder build/compile up front."""
        LOG.info("Preloading Audio8 TTS")
        engine = self._engine
        _synthesize(engine, _WARMUP_SENTENCE, self._resolve_voice(None), self._generation_kwargs,
                    self._decoder_for(engine))

    def _decoder_for(self, engine: Audio8Engine) -> str:
        return _pick_decoder(self.config.get("decoder", "auto"), engine.device)

    def init(self, bus=None, playback=None, callbacks=None):
        # StreamingTTS.init builds subprocess-player callbacks and raises when no ffplay/paplay/aplay
        # is installed. Only require one when streaming playback is actually enabled.
        if not (callbacks or self.config.get("enable_streaming")):
            TTS.init(self, bus, playback)
        else:
            super().init(bus, playback, callbacks)
            if callbacks is None:
                self.callbacks = VisemeStreamCallbacks(self.bus, play_args=_stream_player_args(self.config),
                                                       tts_config=self.config)
        Prewarmer(self._loaded_tensors).register(self.bus)

    def _loaded_tensors(self) -> Optional[list]:
        """Every tensor the next reply will read, or None when the model is not loaded (a wake word must
        not trigger a 2.6 GB load; `preload` does that)."""
        key = (self.config.get("model", DEFAULT_MODEL_ID), self.config.get("revision", DEFAULT_REVISION),
               _pick_device(self.config.get("device", "auto")))
        engine = _engines.get(key)
        if engine is None:
            return None
        modules = [engine.model, engine.model.__dict__.get("_arktts_codec")]
        return [t for m in modules if m is not None for t in [*m.parameters(), *m.buffers()]]

    def viseme(self, phonemes):
        """get_tts puts loudness visemes in the phonemes slot, already as [code, cumulative end] pairs."""
        return decode_visemes(phonemes)

    async def stream_tts(self, sentence: str, lang: str = None, voice: str = None) -> AsyncIterable[bytes]:
        """Yield a streaming WAV (44.1 kHz, 16-bit mono): header first, then PCM as it is decoded.

        The utterance is split into sentences like get_tts does.
        """
        engine = self._engine
        profile, decoder, rate = self._resolve_voice(voice), self._decoder_for(engine), engine.sample_rate
        clock, cost = PlaybackClock(), CostModel()  # shared by every sentence: the player's queue spans them
        clock.add(STREAM_PAD_SECONDS)
        yield wav_stream_header(rate) + silence_pcm16(rate)
        for part in split_for_streaming(sentence):
            for chunk in _stream_sentence(engine, part, profile, self._generation_kwargs, decoder, clock, cost):
                clock.add(len(chunk) / rate)
                clock.speech_written = True
                yield to_pcm16(chunk)
        yield silence_pcm16(rate)

    async def generate_audio(self, sentence, wav_file, play_streaming=True, listen=False, message=None,
                             plugin_kwargs=None):
        """Stream as OVOS does, then fix the placeholder WAV sizes so the cached file is valid."""
        # OPM's queue path creates the cache folder, its streaming path does not; macOS also prunes $TMPDIR.
        os.makedirs(os.path.dirname(os.fspath(wav_file)) or ".", exist_ok=True)
        wav_file = await super().generate_audio(sentence, wav_file, play_streaming, listen, message, plugin_kwargs)
        finalize_wav_header(wav_file)
        return wav_file

    @property
    def _engine(self) -> Audio8Engine:
        return _get_engine(
            self.config.get("model", DEFAULT_MODEL_ID),
            self.config.get("revision", DEFAULT_REVISION),
            _pick_device(self.config.get("device", "auto")),
        )

    def _resolve_voice(self, voice: Optional[str]) -> VoiceProfile:
        # OVOS/Neon pass the literal "default" when no voice was picked.
        if not voice or voice == "default":
            voice = self.config.get("voice")
        if not voice:
            return VoiceProfile(None, None)
        voices = self.config.get("voices", {})
        if voice not in voices:
            raise ValueError(f"Unknown Audio8 voice '{voice}'. Configured voices: {sorted(voices)}")
        entry = voices[voice]
        if entry.get("reference_audio") and not entry.get("reference_text"):
            raise ValueError(f"Audio8 voice '{voice}' needs a reference_text matching its reference_audio")
        return VoiceProfile(entry.get("reference_audio"), entry.get("reference_text"))

    @property
    def _generation_kwargs(self) -> dict:
        return {
            "max_new_tokens": int(self.config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
            "temperature": float(self.config.get("temperature", 0.8)),
            "top_p": float(self.config.get("top_p", 0.95)),
            "top_k": int(self.config.get("top_k", 50)),
        }

    def get_tts(self, sentence: str, wav_file: str, lang: str = None, voice: str = None) -> Tuple[str, str]:
        engine = self._engine
        profile, decoder = self._resolve_voice(voice), self._decoder_for(engine)
        # OVOS only sentence-splits when `sentence_tokenize` is set, so long utterances arrive whole.
        # Generating per sentence keeps each one inside the 2,048-frame context and max_new_tokens.
        parts = split_sentences(sentence)
        audio = np.concatenate([_synthesize(engine, part, profile, self._generation_kwargs, decoder)
                                for part in parts])
        target_rate = int(self.config.get("sample_rate", DEFAULT_SAMPLE_RATE))
        audio = _resample(audio, engine.sample_rate, target_rate)
        _write_pcm16_wav(wav_file, audio, target_rate)
        return wav_file, encode_visemes(loudness_visemes(audio, target_rate))

    def shutdown(self):
        _engines.clear()
        _reference_codes.clear()
        _voice_gains.clear()

    @classmethod
    def available_languages(cls) -> set:
        return {"en"}


Audio8TTSPluginConfig = {
    "en": [
        {
            "voice": "default",
            "lang": "en",
            "meta": {"display_name": "Audio8 TTS — Default", "offline": True, "priority": 55},
        }
    ]
}
