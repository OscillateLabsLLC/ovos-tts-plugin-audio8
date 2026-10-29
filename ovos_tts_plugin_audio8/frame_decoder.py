"""Static-buffer decode loop for Audio8 TTS (batch size 1).

Eager ``ArkttsModel.generate`` is dominated by per-op overhead: each audio frame is ~11
small sequential forward passes plus full-vocabulary sampling, and it syncs with the
host every frame. This loop runs one frame on fixed buffers and then either replays
it as a CUDA graph, runs it through ``torch.compile``, or runs it eagerly.

It also computes semantic logits only for the 4,096 semantic tokens + EOS. Every other
token is masked to -inf before sampling, so the distribution matches ``generate``.

Mirrors ``generate`` at the pinned model revision, using its private helpers
(``_prepare_prompt``, ``_embed``, ``_causal_mask``). Re-verify on revision bumps.
"""

from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ovos_tts_plugin_audio8.chunking import EOS_CHECK_INTERVAL


_WARMUP_FRAMES = 3


class DecodeMode(str, Enum):
    CUDA_GRAPH = "cuda_graph"
    COMPILE = "compile"
    EAGER = "eager"


def _filter_top_k_top_p(scores, top_k: int, top_p: float, temperature: float, positions):
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
    remove_sorted = (cumulative > top_p) | (positions >= top_k)
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return scores.masked_fill(remove, float("-inf")) / max(temperature, 1e-5)


def _sample(scores):
    probabilities = torch.softmax(scores, dim=-1)
    noise = -torch.log(torch.rand_like(probabilities))
    return torch.argmax(probabilities / noise, dim=-1)


class FrameDecoder:
    """Owns the model's generation KV caches and the per-frame step function."""

    def __init__(self, model, top_k: int, top_p: float, temperature: float, mode: DecodeMode):
        if mode is DecodeMode.CUDA_GRAPH and model.device.type != "cuda":
            raise ValueError("CUDA graph decoding requires a CUDA device")
        self.model = model
        self.config = model.config
        self.sampling = (top_k, top_p, temperature)
        self.mode = mode
        # What the caller asked for; differs from `mode` after a compile -> eager fallback.
        self.requested_mode = mode
        # inference_mode everywhere (not no_grad): torch.compile guards on the grad mode it traced under.
        with torch.inference_mode():
            self._allocate_buffers()
            self._step = self._build_step()

    def _allocate_buffers(self):
        cfg, device = self.config, self.model.device
        dtype = next(self.model.parameters()).dtype
        self.model._setup_generation_caches(1, cfg.max_seq_len, dtype)
        self.semantic_ids = torch.cat((
            torch.arange(cfg.semantic_begin_id, cfg.semantic_end_id + 1, device=device),
            torch.tensor([cfg.eos_token_id], device=device),
        ))
        self.semantic_head = self.model.embeddings.weight[self.semantic_ids].contiguous()
        self.semantic_positions = torch.arange(len(self.semantic_ids), device=device)
        self.codebook_positions = torch.arange(cfg.codebook_size, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.step = torch.zeros(1, dtype=torch.long, device=device)
        self.key_mask = torch.ones((1, cfg.max_seq_len), dtype=torch.long, device=device)
        self.logits = torch.zeros((1, len(self.semantic_ids)), dtype=dtype, device=device)
        self.slow_hidden = torch.zeros((1, 1, cfg.dim), dtype=dtype, device=device)
        self.previous = torch.full((1, cfg.ras_window_size), -1, dtype=torch.long, device=device)
        self.semantic_out = torch.zeros(cfg.max_seq_len, dtype=torch.long, device=device)
        self.codes_out = torch.zeros((cfg.num_codebooks, cfg.max_seq_len), dtype=torch.long, device=device)
        self.fast_positions = [torch.tensor([i], device=device) for i in range(cfg.num_codebooks)]
        fast_keys = torch.ones((1, cfg.num_codebooks), dtype=torch.bool, device=device)
        self.fast_masks = [
            self.model._causal_mask(fast_keys, position, cfg.num_codebooks) for position in self.fast_positions
        ]

    def _build_step(self):
        if self.mode is DecodeMode.CUDA_GRAPH:
            return self._capture_graph().replay
        step = torch.compile(self._frame, dynamic=False) if self.mode is DecodeMode.COMPILE else self._frame
        for _ in range(_WARMUP_FRAMES):
            step()
        return step

    def _capture_graph(self) -> torch.cuda.CUDAGraph:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_FRAMES):
                self._frame()
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._frame()
        return graph

    def _frame(self):
        semantic = self._sample_semantic()
        codebooks = self._codebooks(semantic)
        self.semantic_out.index_copy_(0, self.step, semantic)
        self.codes_out.index_copy_(1, self.step, codebooks.T)
        column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)
        with sdpa_kernel(SDPBackend.MATH):
            logits, hidden = self._slow_forward(column, self.position, self.position[:, None], self.key_mask)
        self.logits.copy_(logits)
        self.slow_hidden.copy_(hidden)
        self.position += 1
        self.step += 1

    def _slow_forward(self, ids, cache_position, position_ids, attention_mask):
        model = self.model
        hidden = model._embed(ids)
        rope = model.freqs_cis[position_ids]
        mask = model._causal_mask(attention_mask, cache_position, self.config.max_seq_len)
        for layer in model.layers:
            hidden = layer(hidden, rope, mask, cache_position)
        hidden = hidden[:, -1:]
        normalized = model.norm(hidden)
        logits = F.linear(normalized[:, -1], self.semantic_head)
        return logits, normalized if self.config.norm_fastlayer_input else hidden

    def _sample_semantic(self):
        cfg = self.config
        top_k, top_p, temperature = self.sampling
        positions = self.semantic_positions
        normal = self.semantic_ids[_sample(_filter_top_k_top_p(self.logits, top_k, top_p, temperature, positions))]
        # Repetition-aware sampling: resample with RAS settings if the token repeats recently.
        high_scores = _filter_top_k_top_p(self.logits, top_k, cfg.ras_top_p, cfg.ras_temperature, positions)
        high = self.semantic_ids[_sample(high_scores)]
        repeated = (self.previous == normal[:, None]).any(dim=1)
        # Nested where instead of `repeated & not_eos`: inductor's MPS codegen fails on bool bitwise_and.
        semantic = torch.where(repeated, torch.where(normal != cfg.eos_token_id, high, normal), normal)
        self.previous.copy_(torch.cat((self.previous[:, 1:], semantic[:, None]), dim=1))
        return semantic

    def _codebooks(self, semantic):
        model, cfg = self.model, self.config
        top_k, top_p, temperature = self.sampling
        self._fast_step(model.fast_project_in(self.slow_hidden), 0)
        current = (semantic - cfg.semantic_begin_id).clamp(0, cfg.codebook_size - 1)
        codebooks = [current]
        for position in range(1, cfg.num_codebooks):
            scores = self._fast_step(model.fast_embeddings(current)[:, None], position)
            current = _sample(_filter_top_k_top_p(scores, top_k, top_p, temperature, self.codebook_positions))
            codebooks.append(current)
        return torch.stack(codebooks, dim=1)

    def _fast_step(self, hidden, position: int):
        model = self.model
        cache_position = self.fast_positions[position]
        rope = model.fast_freqs_cis[cache_position]
        for layer in model.fast_layers:
            hidden = layer(hidden, rope, self.fast_masks[position], cache_position)
        return model.fast_output(model.fast_norm(hidden))[:, -1]

    def generate(self, inputs: dict, max_new_tokens: int) -> torch.Tensor:
        """Run prefill eagerly, then step frames until EOS. Returns codes [1, num_codebooks, T]."""
        frames = 0
        for frames in self.iter_generate(inputs, max_new_tokens):
            pass
        return self.codes_out[:, :frames].clone().unsqueeze(0)

    @torch.inference_mode()
    def iter_generate(self, inputs: dict, max_new_tokens: int):
        """Yield the number of valid frames in ``codes_out`` after each EOS check; the last value is final.

        ``codes_out`` is overwritten by the next utterance, so consumers must copy what they keep.
        """
        prompt, prompt_mask = self.model._prepare_prompt(**inputs)
        width = prompt.shape[-1]
        budget = min(max_new_tokens, self.config.max_seq_len - width)
        if budget <= 0:
            raise ValueError(f"Prompt length {width} leaves no room in the {self.config.max_seq_len} context")
        positions = torch.arange(width, device=self.model.device)
        logits, hidden = self._slow_forward(prompt, positions, positions[None], prompt_mask)
        self.logits.copy_(logits)
        self.slow_hidden.copy_(hidden)
        self.position.fill_(width)
        self.step.zero_()
        self.previous.fill_(-1)
        yield from self._run_until_eos(budget)

    def _run_until_eos(self, budget: int):
        steps = 0
        while steps < budget:
            batch = min(EOS_CHECK_INTERVAL, budget - steps)
            for _ in range(batch):
                self._step()
            steps += batch
            eos = (self.semantic_out[:steps] == self.config.eos_token_id).nonzero()
            if eos.numel():
                yield int(eos[0])
                return
            yield steps
