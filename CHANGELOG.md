# Changelog

## 0.1.0 (2026-10-29)


### Features

* Audio8 TTS plugin with one-shot voice cloning and fast decoders ([67c3c9e](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/67c3c9e01673029ef851c88bf61be09d291c8066))
* Audio8 TTS plugin with one-shot voice cloning, fast decoders and streaming ([c9e67e0](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/c9e67e0d578c95e136e5a368d7bc215e66336e13))
* even out loudness across cloned voices ([8fe0e0a](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/8fe0e0a2599a25b53cba9e20ebfc31750dcd880a))
* loudness-driven visemes for the GUI mouth ([3a16e02](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/3a16e024b95e98992a9c9504a7f4ce89e9148172))
* stream audio while frames generate (OVOS StreamingTTS) ([8120c64](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/8120c64c1909320c1b3dc2c76deeba4950939310))


### Bug Fixes

* drop the junk final frame and pad streams with silence ([4a833e7](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/4a833e7b8ed20582b35a6dc9961919909e380eb7))
* keep ffplay fed through the stream ramp-up ([790d7f1](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/790d7f1824c548c9f564df7a0118bcae2e5a859b))
* play streamed audio with low-latency ffplay flags ([d58c9f8](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/d58c9f84d6bb7854c5da868823ed51ee67667068))
* prewarm on the wake word and stream long sentences in clauses ([093c001](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/093c001e900cdf8690bae799cd1352336d002957))
* recreate the TTS cache folder before streaming into it ([9e3db51](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/9e3db514f38bb79131bb6e28729eb744eddaadf9))
* restore RoPE buffers so transformers 5 generates speech ([4b66023](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/4b6602381d03497e7af3c672ab2df32285237615))
* schedule streamed chunks by playback deadline, not doubling ([bc24fd1](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/bc24fd186a8732ea000a9205fdf30cbd3561a929))


### Documentation

* describe streaming scheduler, loudness, warm-up and known ffplay limit ([e64bee9](https://github.com/OscillateLabsLLC/ovos-tts-plugin-audio8/commit/e64bee9634f55620949fd154da5acafcd93aab52))
