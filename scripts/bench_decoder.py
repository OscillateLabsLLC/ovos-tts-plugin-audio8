"""Compare decode modes on the same phrases and save every output for listening/ASR.

Modes: model (upstream generate()), eager, compile, cuda_graph.
Usage: uv run python scripts/bench_decoder.py --device mps --modes model,eager,compile --out bench
"""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from generate_samples import PHRASES, REFERENCE_AUDIO, REFERENCE_TEXT

from ovos_tts_plugin_audio8 import tts
from ovos_tts_plugin_audio8.frame_decoder import DecodeMode, FrameDecoder


SAMPLING = {"top_k": 50, "top_p": 0.95, "temperature": 0.8}
MAX_NEW_TOKENS = 1024


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _inputs(engine, phrase):
    voice = tts.VoiceProfile(str(REFERENCE_AUDIO), REFERENCE_TEXT)
    return tts._build_inputs(engine, phrase, voice)


def _producer(engine, mode: str):
    if mode == "model":
        def generate(inputs):
            output = engine.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                           return_dict_in_generate=True, **SAMPLING)
            return output.codes
        return generate
    started = time.time()
    decoder = FrameDecoder(engine.model, mode=DecodeMode(mode), **SAMPLING)
    print(f"{mode}: decoder ready in {time.time() - started:.1f}s", flush=True)
    return lambda inputs: decoder.generate(inputs, MAX_NEW_TOKENS)


def _timed(engine, mode, produce, phrase, out_path):
    _sync(engine.device)
    started = time.time()
    with torch.inference_mode():
        codes = produce(_inputs(engine, phrase))
        waveforms, lengths = engine.model.decode_audio(codes)
    _sync(engine.device)
    elapsed = time.time() - started
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    sf.write(out_path, audio, engine.sample_rate)
    seconds = len(audio) / engine.sample_rate
    print(f"{mode:10s} {out_path.name}  {seconds:5.2f}s audio  {elapsed:5.2f}s wall  RTF {elapsed / seconds:.3f}",
          flush=True)
    return elapsed, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--modes", default="model,eager,compile")
    parser.add_argument("--out", default="bench")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = tts._get_engine(tts.DEFAULT_MODEL_ID, tts.DEFAULT_REVISION, args.device)

    for mode in args.modes.split(","):
        produce = _producer(engine, mode)
        _timed(engine, mode, produce, PHRASES[0], out_dir / f"{mode}_warmup.wav")
        totals = [_timed(engine, mode, produce, phrase, out_dir / f"{mode}_{index:02d}.wav")
                  for index, phrase in enumerate(PHRASES, start=1)]
        wall, audio = (sum(column) for column in zip(*totals))
        print(f"== {mode}: aggregate RTF {wall / audio:.3f}", flush=True)


if __name__ == "__main__":
    main()
