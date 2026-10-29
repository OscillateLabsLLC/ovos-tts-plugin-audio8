import threading
from unittest.mock import MagicMock, patch

import torch

from ovos_tts_plugin_audio8 import Audio8TTSPlugin, prewarm, tts


def test_touch_reads_every_tensor():
    assert prewarm.touch_tensors([torch.ones(3), torch.zeros(0), torch.arange(4)]) == 2


def test_trigger_runs_once_per_interval():
    now = [100.0]
    calls = []
    warmer = prewarm.Prewarmer(lambda: calls.append(1) or [torch.ones(2)], clock=lambda: now[0])
    assert warmer.trigger()
    warmer.thread.join(1)
    now[0] = 101.0
    assert not warmer.trigger()  # one second later: too soon
    now[0] = 130.0
    assert warmer.trigger()
    warmer.thread.join(1)
    assert len(calls) == 2


def test_a_warmup_in_progress_is_not_doubled():
    release = threading.Event()
    warmer = prewarm.Prewarmer(lambda: release.wait(1) and [torch.ones(2)], clock=iter(range(0, 1000, 50)).__next__)
    assert warmer.trigger()
    assert not warmer.trigger()  # 50 s later but the first is still running
    release.set()
    warmer.thread.join(1)


def test_unloaded_model_is_not_loaded_by_a_wake_word():
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False})
    with patch.dict(tts._engines, clear=True):
        assert plugin._loaded_tensors() is None


def test_init_listens_for_the_wake_word():
    plugin = Audio8TTSPlugin(config={"lang": "en-us", "preload": False})
    bus = MagicMock()
    with patch.object(tts.TTS, "playback", None):
        plugin.init(bus=bus, playback=MagicMock())
    registered = {call.args[0] for call in bus.on.call_args_list}
    assert set(prewarm.PREWARM_EVENTS) <= registered
