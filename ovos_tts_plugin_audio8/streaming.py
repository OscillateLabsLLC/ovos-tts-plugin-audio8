"""Incremental audio for Audio8: decode codec chunks while frames are still being generated.

The codec is causal, so decoding a *prefix* of the codes reproduces the full-utterance samples
exactly (measured to float noise on CPU and MPS). A bounded left-context window does not:
its stacked windowed attention sees the whole utterance (error still 1e-3 with 256 frames of
context). So every chunk re-decodes from the sentence start and emits only the new samples;
chunking.py decides when.
"""

import struct
import time
from typing import Iterator

import numpy as np

from ovos_tts_plugin_audio8.chunking import EOS_CHECK_INTERVAL, CostModel, PlaybackClock, should_emit


# The model's last frame before EOS decodes to a rising burst (-20 dB after near-silence) in every
# sentence measured, so it is never played; streaming holds the newest frame back until another follows.
TRAILING_JUNK_FRAMES = 1
# ffplay drops roughly the first 100-140 ms of a piped stream and can clip the last ones at -autoexit,
# so the stream opens and closes on silence.
STREAM_PAD_SECONDS = 0.2
_STREAMING_SIZE = 0xFFFFFFFF
_RIFF_HEADER_BYTES = 44


def wav_stream_header(sample_rate: int) -> bytes:
    """Mono 16-bit PCM WAV header with unknown length, as players expect for streams."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", _STREAMING_SIZE, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", _STREAMING_SIZE,
    )


def finalize_wav_header(path: str):
    """Replace the streaming placeholder sizes once the whole file is on disk."""
    with open(path, "r+b") as wav:
        wav.seek(0, 2)
        size = wav.tell()
        if size < _RIFF_HEADER_BYTES:
            raise ValueError(f"{path} is too short to be a WAV file ({size} bytes)")
        wav.seek(4)
        wav.write(struct.pack("<I", size - 8))
        wav.seek(40)
        wav.write(struct.pack("<I", size - _RIFF_HEADER_BYTES))


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def silence_pcm16(sample_rate: int, seconds: float = STREAM_PAD_SECONDS) -> bytes:
    return bytes(2 * int(sample_rate * seconds))


def playable_frames(frames: int) -> int:
    return max(0, frames - TRAILING_JUNK_FRAMES)


def stream_codes_to_audio(model, decoder, inputs: dict, max_new_tokens: int, clock: PlaybackClock,
                          cost: CostModel) -> Iterator[np.ndarray]:
    """Yield float32 audio chunks at the codec rate, each just before the player would run dry.

    The caller adds every chunk it writes (including silence) to `clock`.
    """
    frame_samples = int(model.config.codec_frame_size)
    frame_seconds = frame_samples / int(model.config.codec_sample_rate)
    emitted = frames = generated = 0
    resumed = time.monotonic()
    for frames in decoder.iter_generate(inputs, max_new_tokens):
        cost.observe_generation(frames - generated, time.monotonic() - resumed)
        generated, ready = frames, playable_frames(frames)
        if should_emit(ready - emitted, ready, EOS_CHECK_INTERVAL, frame_seconds, clock, cost):
            yield _decode_new_frames(model, decoder.codes_out, emitted, ready, frame_samples, cost)
            emitted = ready
        resumed = time.monotonic()
    final = playable_frames(frames)
    if final > emitted:
        yield _decode_new_frames(model, decoder.codes_out, emitted, final, frame_samples, cost)


def _decode_new_frames(model, codes, emitted: int, frames: int, frame_samples: int, cost: CostModel) -> np.ndarray:
    started = time.monotonic()
    waveforms, lengths = model.decode_audio(codes[:, :frames].clone().unsqueeze(0))
    audio = waveforms[0, emitted * frame_samples: int(lengths[0])].float().cpu().numpy()
    cost.observe_decode(frames, time.monotonic() - started)  # .cpu() synchronizes, so this is real decode time
    return audio
