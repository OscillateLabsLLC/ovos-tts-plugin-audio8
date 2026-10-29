"""When to decode the next streamed chunk: as late as possible without starving the player.

Every chunk re-decodes its sentence from the first frame (see streaming.py), so each decode costs more
as the sentence grows. Fixed growth (doubling) left the player empty before every chunk on long
sentences at MPS speed (1.5 s of gaps over one 10 s sentence). Instead, at each EOS check the stream
decodes only when waiting for the next check would let the queued audio run out.
"""

import time
from typing import Callable


# Frames the decoder runs between host syncs that look for EOS (frames past EOS are discarded); the
# stream re-plans at each of these checks. Lives here, not in frame_decoder, so streaming never imports torch.
EOS_CHECK_INTERVAL = 8
# ffplay with low-latency flags: playback starts ~0.2 s after the first write and drops ~0.1 s.
STREAM_PLAYBACK_LATENCY = 0.1
# Audio kept queued beyond the estimated cost of waiting. ffplay discards ~0.4 s and resyncs when a chunk
# lands as its buffer runs dry (heard as a clip; measured played-vs-sent), so stay well clear of empty.
SAFETY_SECONDS = 0.5
MIN_CHUNK_FRAMES = 6
# The utterance's first chunk is the only lead the player has while the stream ramps up. On MPS, 15
# frames (0.70 s) still ran dry ~2 s in, where ffplay then discarded audio (heard as a clip on the
# weather line). 31 frames (1.44 s; checks land on 8k-1) left no gap inside speech on the measured lines,
# at ~1.0 s to first audio instead of ~0.56 s.
STARTUP_CHUNK_FRAMES = 31
CHUNK_MARGIN = 1.5
_SMOOTHING = 0.5
# Starting estimates from M3 Max MPS measurements; replaced by observations within a few checks.
_INITIAL_SECONDS_PER_GENERATED_FRAME = 0.025
_INITIAL_SECONDS_PER_DECODED_FRAME = 0.005


class PlaybackClock:
    """Tracks when everything written to the player so far finishes playing."""

    def __init__(self, latency: float = STREAM_PLAYBACK_LATENCY, now: Callable[[], float] = time.monotonic):
        self.latency = latency
        self.now = now
        self.playhead = 0.0
        self.speech_written = False  # set by the stream once real audio (not padding) has gone out

    def queued(self) -> float:
        """Seconds of written audio still waiting to play."""
        return max(0.0, self.playhead - (self.now() + self.latency))

    def add(self, seconds: float) -> float:
        """Record a written chunk; a chunk written after the queue ran dry starts on arrival. Returns its start."""
        start = max(self.playhead, self.now() + self.latency)
        self.playhead = start + seconds
        return start


class CostModel:
    """Running estimates of seconds per generated frame and per frame of prefix decode."""

    def __init__(self):
        self.generate = _INITIAL_SECONDS_PER_GENERATED_FRAME
        self.decode = _INITIAL_SECONDS_PER_DECODED_FRAME

    def observe_generation(self, frames: int, seconds: float):
        if frames > 0:
            self.generate += _SMOOTHING * (seconds / frames - self.generate)

    def observe_decode(self, frames: int, seconds: float):
        if frames > 0:
            self.decode += _SMOOTHING * (seconds / frames - self.decode)


def sustainable_chunk(ready: int, frame_seconds: float, cost: CostModel) -> float:
    """Smallest chunk whose audio outlasts the time to generate it plus re-decode the whole prefix.

    Emitting less than this falls further behind with every chunk (measured: 8-frame chunks on a 186-frame
    MPS sentence cost 0.8 s of decode for 0.37 s of audio). Margin 1.5x.
    """
    spare = frame_seconds - cost.generate
    if spare <= 0:
        return float("inf")  # generation alone is slower than real time: one decode at the end is cheapest
    return CHUNK_MARGIN * cost.decode * ready / spare


def should_emit(pending: int, ready: int, check_frames: int, frame_seconds: float, clock: PlaybackClock,
                cost: CostModel) -> bool:
    """Decode now if the chunk is sustainable and waiting for `check_frames` more would let the player run dry."""
    smallest = MIN_CHUNK_FRAMES if clock.speech_written else STARTUP_CHUNK_FRAMES
    if pending < max(smallest, sustainable_chunk(ready, frame_seconds, cost)):
        return False
    cost_of_waiting = cost.generate * check_frames + cost.decode * (ready + check_frames) + SAFETY_SECONDS
    return clock.queued() < cost_of_waiting
