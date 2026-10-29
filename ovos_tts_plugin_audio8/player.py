"""Built-in stream player: plays silence when it runs dry instead of discarding audio.

ffplay reading stdin dropped ~0.4 s and resynced when a chunk landed as its buffer emptied (heard as
a clip; measured played-vs-sent), dropped the first ~0.1 s of a stream, and clipped the end. This
player owns the buffer: an underrun is a gap of silence, nothing written is ever skipped, and the
device's reported latency replaces a guess.
"""

import threading
import time
from typing import Callable, Optional

from ovos_utils.log import LOG


SAMPLE_BYTES = 2  # mono int16
# Fixed callback size: without it CoreAudio called back every ~15 frames (891 calls in a 0.3 s stall).
BLOCK_FRAMES = 512
_DRAIN_POLL_SECONDS = 0.01


def sounddevice_stream(sample_rate: int, callback):
    """Open a low-latency mono int16 output stream on the default device."""
    import sounddevice

    return sounddevice.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16", latency="low",
                                       blocksize=BLOCK_FRAMES, callback=callback)


class PcmPlayer:
    """Plays mono 16-bit PCM written in arbitrary chunks, in order, with no drops."""

    def __init__(self, sample_rate: int, stream_factory: Callable = sounddevice_stream):
        self.sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._playing = False  # True once audio has been written; underruns before that are just startup
        self._starved = False
        self._finishing = False  # set by drain(): running out at the end is expected, not an underrun
        self.underruns = 0  # times playback ran dry mid-stream (events, not callbacks)
        self.samples_played = 0
        self._stream = stream_factory(sample_rate, self._callback)
        self.latency: float = float(getattr(self._stream, "latency", 0.0) or 0.0)
        self._stream.start()

    def write(self, pcm: bytes):
        with self._lock:
            self._buffer.extend(pcm)
            self._playing = True

    def _callback(self, outdata, frames: int, time_info, status):
        wanted = frames * SAMPLE_BYTES
        with self._lock:
            take = bytes(self._buffer[:wanted])
            del self._buffer[:wanted]
            short = len(take) < wanted
            if short and self._playing and not self._finishing and not self._starved:
                self.underruns += 1
            self._starved = short and self._playing
        outdata[: len(take)] = take
        if len(take) < wanted:
            outdata[len(take):] = bytes(wanted - len(take))
        self.samples_played += frames

    def buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / SAMPLE_BYTES / self.sample_rate

    def drain(self, timeout: Optional[float] = None):
        """Block until everything written has been handed to the device and played out, then close."""
        deadline = None if timeout is None else time.monotonic() + timeout
        self._finishing = True
        while self.buffered_seconds() > 0:
            if deadline is not None and time.monotonic() > deadline:
                LOG.warning("Audio8 player drain timed out with %.2f s unplayed", self.buffered_seconds())
                break
            time.sleep(_DRAIN_POLL_SECONDS)
        time.sleep(self.latency)  # the device still holds this much after the last callback
        self.close()

    def close(self):
        self._stream.stop()
        self._stream.close()
