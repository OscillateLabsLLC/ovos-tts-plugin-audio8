# ovos-tts-plugin-audio8

[![Status: Proof of Concept](https://img.shields.io/badge/status-proof%20of%20concept-orange)](https://github.com/OscillateLabsLLC/.github/blob/main/SUPPORT_STATUS.md)

> **Proof of concept: experimental, not for production, may be abandoned.** The API can change.

OVOS TTS plugin for [Audio8 TTS Realistic 0.6B](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b).
It clones a voice from one short reference clip. The plugin replaces the model's decode loop with a
faster one, so it runs faster than real time on an NVIDIA GPU or an Apple Silicon Mac.

## Hardware: read this first

This plugin needs a GPU. **It is not for a Raspberry Pi** or other small single-board computers.

| Hardware | Real time? | Steady RTF | Memory |
| :--- | :--- | :--- | :--- |
| NVIDIA GPU (CUDA), tested on RTX 3090 | **Yes** | **0.27** | ~3 GB VRAM |
| Apple Silicon (MPS), tested on M3 Max | **Yes** | **0.54** | ~4.7 GB unified memory |
| CPU only, tested on M3 Max | No | 1.49 | ~5.3 GB RAM |
| Raspberry Pi / ARM SBC | No | not supported | — |

RTF is synthesis time divided by audio length. A value below 1.0 is faster than real time. At RTF 0.27,
a 5-second sentence takes about 1.4 s to make.

If your voice assistant runs on small hardware, run this plugin on a machine that has a GPU, behind
[ovos-tts-server](https://github.com/OpenVoiceOS/ovos-tts-server), and point your assistant at that
server.

## Install

```bash
pip install ovos-tts-plugin-audio8
# For streaming, add the built-in player:
pip install "ovos-tts-plugin-audio8[player]"
```

- On Linux, pip installs the CUDA build of PyTorch by default.
- On macOS, the default PyTorch build includes MPS support.
- The first start downloads about 2.6 GB of model files from Hugging Face.

## Configure

```json
"tts": {
  "module": "ovos-tts-plugin-audio8",
  "ovos-tts-plugin-audio8": {
    "voice": "seven",
    "voices": {
      "seven": {
        "reference_audio": "/home/ovos/voices/regeneration_cycle.wav",
        "reference_text": "Oh six hundred hours, regeneration cycle complete."
      }
    }
  }
}
```

The defaults select the fastest settings for your hardware. You must set only `voice` and `voices`.

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `voice` | none | Name of an entry in `voices`. If you do not set it, the model uses its own default voice. |
| `voices` | `{}` | Map of voice name to `reference_audio` and `reference_text`. |
| `device` | `auto` | `cuda`, `mps` or `cpu`. `auto` selects CUDA, then MPS, then CPU. |
| `decoder` | `auto` | `cuda_graph`, `compile`, `eager` or `model`. `auto` selects the fastest decoder for the device. Refer to [Performance](#performance). |
| `preload` | `true` | Load the model and build the decoder when the plugin starts. This takes 10–60 s, so the first sentence is fast. |
| `sample_rate` | `22050` | Output WAV rate. The model makes 44.1 kHz audio; the plugin resamples it. |
| `temperature`, `top_p`, `top_k` | `0.8`, `0.95`, `50` | Sampling settings. |
| `max_new_tokens` | `1024` | Frame limit for **one sentence**, about 47 s of audio. The plugin splits long utterances into sentences, so there is no limit on total length. This limit stops a sentence that never ends. |
| `enable_streaming` | `false` | Play audio while the sentence is still being generated. Refer to [Streaming](#streaming). |
| `stream_player` | `auto` | `sounddevice`, `ffplay` or `auto`. `auto` uses the built-in player when `sounddevice` is installed and there is an output device, and `ffplay` if not. |
| `stream_player_args` | low-latency `ffplay` | Command that plays the stream from stdin, for example `["paplay", "-"]`. If you set it, `auto` uses this command. |
| `model`, `revision` | pinned | Hugging Face repository and commit. |

## Voice cloning

- `reference_text` must be the **exact** words in the clip. Write numbers the way the speaker says
  them ("oh six hundred", not "0600"). If the text does not match, the voice is less stable.
- Use a clean clip of 3 to 10 seconds. A 44.1 kHz mono WAV prevents resampling:
  `ffmpeg -i clip.mp3 -ac 1 -ar 44100 clip.wav`.
- To get the transcript, you can use Whisper. Tell it to write numbers as words.
- The plugin encodes each reference clip one time and keeps the codes in memory.
- The reference, its transcript and the sentence share a 2,048-frame context. Use short references.
- Each take is sampled, so takes of the same sentence are different. A lower `temperature` gives
  takes that are more alike.
- The model copies the loudness of the clip. The plugin measures each clip and applies one gain per
  voice, so all voices play at about the same level (within ~4 dB in tests; a quiet clip and a loud
  clip were 16 dB apart before).

Only clone a voice that you have the right to use.

## Streaming

The plugin is an OVOS `StreamingTTS`. Set `"enable_streaming": true`, and `ovos-audio` starts to
play each sentence before the plugin has finished generating it.

| M3 Max, MPS | Time to first audio, 6.5 s utterance |
| :--- | :--- |
| Without streaming | ~2.7 s (first sentence fully generated) |
| With streaming | **~1.0 s** |

- The built-in player (the `player` extra) plays the stream through `sounddevice`. When a chunk is
  late, it plays silence and then continues. It does not discard audio.
- Without the extra, streaming needs `ffplay`, `paplay` or `aplay` on the device. The plugin starts
  `ffplay` with low-latency flags; the default flags wait for about 2 s of audio before they play.
  When `ffplay` runs out of audio, it discards some audio, so a late chunk can cause a short glitch.
- Streamed audio is 44.1 kHz, the codec rate. The `sample_rate` setting does not apply.
- Each chunk decodes the sentence again from its first frame. The codec looks back over the whole
  sentence, so this is the only exact way to decode in chunks. A scheduler sends each chunk just
  before the player would run out, and never a chunk too small to pay for its own decode. The first
  chunk is about 1.4 s of audio, so playback starts with a lead.
- Long sentences (more than 100 characters) are split at commas into pieces of 40 to 100 characters,
  so that each decode stays short. Without streaming, sentences are not split.
- When the wake word is heard, the plugin reads the whole model once, in the background. This
  removes a ~2 s delay on the first reply after the device has been idle.
- Mouth movement: the model gives no phoneme timing, so the plugin sends visemes from the audio level
  in 40 ms steps (closed, half open, round, wide). With streaming, each chunk's visemes go out when that
  chunk starts to play. Without streaming, the visemes travel with the WAV and are kept in the OVOS cache.
- Streaming works only when the plugin runs inside `ovos-audio`. The Neon Hub (`neon-audio`) and
  `ovos-tts-server` 1.15 always request complete files.

## Performance

Steady RTF for each `decoder` value. The test used 8 sentences and a cloned voice.

| Hardware | `model` | `eager` | `compile` | `cuda_graph` |
| :--- | :--- | :--- | :--- | :--- |
| RTX 3090, bf16 | 1.26 | — | — | **0.27** |
| M3 Max, MPS, fp32 | 1.28 | 0.68 | **0.54** | n/a |
| M3 Max, CPU, fp32 | 2.14 | 1.98 | **1.49** | n/a |

`auto` selects `cuda_graph` on CUDA and `compile` on other devices. If `torch.compile` fails, the
plugin logs the error and uses `eager`.

### Why it is faster

The model's own `generate()` is limited by per-operation overhead, not by computation. Each audio
frame runs about 11 small forward passes in sequence: one pass of the 24-layer model, then 10 steps of
a 4-layer codebook model. On a 3090, GPU use stays near 30%. On MPS, the codebook loop uses about two
thirds of each frame.

`frame_decoder.py` runs each frame on fixed buffers:

- It does not sync with the host on each frame. It checks for end of speech every 8 frames.
- It does not create tensors from Python values inside the loop.
- It computes logits and samples only over the 4,096 semantic tokens and EOS. The model masks all
  other tokens before sampling, so the sampled distribution does not change.

Then it runs the frame in one of three ways:

- `cuda_graph` records one full frame as a CUDA graph and replays it.
- `compile` fuses the frame with `torch.compile`.
- `eager` runs it as normal PyTorch.

One difference from upstream: the first frame's token enters the repetition-aware sampling window.
Upstream skips it.

## Security

The model loads custom Python code from Hugging Face (`trust_remote_code=True`). The plugin pins
`revision` to a reviewed commit. `frame_decoder.py` also calls private methods of that code. Before
you change `revision`, read the `*.py` files at the new commit, then run
`scripts/bench_decoder.py` and listen to the output.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

The tests build a small model with random weights from the pinned upstream code. They download the
config and code, but not the weights.

To render sample sentences with your own reference clip, put it at `voices/regeneration_cycle.wav`
and run:

```bash
uv run python scripts/generate_samples.py
uv run python scripts/bench_decoder.py --device mps --modes model,eager,compile
```

Git ignores `voices/*.wav` and `samples/`, because reference clips can be copyrighted audio.

## License

Apache-2.0. The Audio8 model is Apache-2.0, by [Edge0](https://huggingface.co/Edge0).
