"""Wake the model before a reply needs it.

After a few idle minutes on a Mac under memory pressure, the first reply started ~2 s late and had a
0.9 s gap; an immediate second reply started in 0.8 s with 0.08 s gaps. The GPU had clocked down and/or
macOS had paged the idle model out. The wake word gives 1-3 s of warning while STT and intent matching
run, so reading every weight and buffer once then brings it all back.

Read-only and stateless (no KV cache writes), so it is safe beside a synthesis that is already running.
"""

import threading
import time
from typing import Callable, Iterable, Optional

from ovos_utils.log import LOG


PREWARM_EVENTS = ("recognizer_loop:wakeword", "recognizer_loop:record_begin", "mycroft.mic.listen")
MIN_INTERVAL_SECONDS = 20.0


def touch_tensors(tensors: Iterable) -> int:
    """Read every element of every tensor on its device, then synchronize. Returns the number touched."""
    import torch

    sums = []
    with torch.inference_mode():
        for tensor in tensors:
            if tensor.numel():
                sums.append(tensor.sum(dtype=torch.float32))
        if sums:
            torch.stack(sums).sum().item()  # .item() waits for the device to finish every read
    return len(sums)


class Prewarmer:
    """Runs `touch` on a background thread when OVOS starts listening, at most once per interval."""

    def __init__(self, tensors: Callable[[], Optional[list]], clock: Callable[[], float] = time.monotonic):
        self._tensors = tensors
        self._clock = clock
        self._lock = threading.Lock()
        self._last = float("-inf")
        self.thread: Optional[threading.Thread] = None

    def register(self, bus):
        for event in PREWARM_EVENTS:
            bus.on(event, self.trigger)

    def trigger(self, message=None) -> bool:
        """Start a warm-up unless one ran recently or is running. Returns whether it started."""
        now = self._clock()
        if now - self._last < MIN_INTERVAL_SECONDS or not self._lock.acquire(blocking=False):
            return False
        self._last = now
        self.thread = threading.Thread(target=self._run, daemon=True, name="audio8-prewarm")
        self.thread.start()
        return True

    def _run(self):
        try:
            tensors = self._tensors()
            if tensors is None:
                return  # model not loaded yet; nothing to wake
            started = self._clock()
            count = touch_tensors(tensors)
            LOG.debug("Audio8 prewarm touched %d tensors in %.0f ms", count, (self._clock() - started) * 1000)
        except Exception:
            LOG.exception("Audio8 prewarm failed")
        finally:
            self._lock.release()
