"""FrameDecoder against a tiny random-weight Audio8 model built from the pinned upstream code.

Downloads only config + *.py from the Hub (no weights), so it runs on CPU in CI.
"""

import pytest
import torch

from ovos_tts_plugin_audio8 import tts
from ovos_tts_plugin_audio8.frame_decoder import DecodeMode, FrameDecoder


TINY = {
    "dim": 32, "fast_dim": 32, "n_layer": 2, "n_fast_layer": 1, "n_head": 2, "n_local_heads": 1,
    "fast_n_head": 2, "fast_n_local_heads": 1, "head_dim": 16, "fast_head_dim": 16,
    "intermediate_size": 64, "fast_intermediate_size": 64, "max_seq_len": 256,
}
SAMPLING = (50, 0.95, 0.8)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(tts.DEFAULT_MODEL_ID, revision=tts.DEFAULT_REVISION, trust_remote_code=True)
    for key, value in TINY.items():
        setattr(config, key, value)
    torch.manual_seed(0)
    return AutoModel.from_config(config, trust_remote_code=True, dtype=torch.float32).eval()


@pytest.fixture
def decoder(model):
    return FrameDecoder(model, *SAMPLING, DecodeMode.EAGER)


def _prompt(model, width=12):
    ids = torch.zeros((1, model.config.num_codebooks + 1, width), dtype=torch.long)
    ids[:, 0] = torch.randint(0, 1000, (width,))
    return ids


@torch.inference_mode()
def test_sliced_semantic_logits_match_full_vocab(model, decoder):
    prompt = _prompt(model)
    width = prompt.shape[-1]
    positions = torch.arange(width)
    mask = torch.ones((1, width), dtype=torch.long)
    sliced, sliced_hidden = decoder._slow_forward(prompt, positions, positions[None], mask)
    full, full_hidden = model._slow_step(prompt, positions, positions[None], mask)
    torch.testing.assert_close(sliced, full[:, decoder.semantic_ids])
    torch.testing.assert_close(sliced_hidden, full_hidden)


@torch.inference_mode()
def test_semantic_sampling_only_emits_semantic_tokens_or_eos(model, decoder):
    cfg = model.config
    for _ in range(50):
        decoder.logits.copy_(torch.randn_like(decoder.logits) * 5)
        token = int(decoder._sample_semantic())
        assert token == cfg.eos_token_id or cfg.semantic_begin_id <= token <= cfg.semantic_end_id


@torch.inference_mode()
def test_generate_returns_valid_codes_within_budget(model, decoder):
    prefix = torch.randint(0, 1000, (1, 8))
    suffix = torch.randint(0, 1000, (1, 4))
    inputs = {
        "prefix_input_ids": prefix, "prefix_attention_mask": torch.ones_like(prefix),
        "suffix_input_ids": suffix, "suffix_attention_mask": torch.ones_like(suffix),
    }
    codes = decoder.generate(inputs, max_new_tokens=20)
    assert codes.shape[:2] == (1, model.config.num_codebooks)
    assert codes.shape[2] <= 20
    assert codes.numel() == 0 or (codes.min() >= 0 and codes.max() < model.config.codebook_size)


def test_cuda_graph_mode_rejects_non_cuda_device(model):
    with pytest.raises(ValueError, match="CUDA"):
        FrameDecoder(model, *SAMPLING, DecodeMode.CUDA_GRAPH)
