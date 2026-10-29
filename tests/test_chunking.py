from ovos_tts_plugin_audio8.chunking import CostModel, PlaybackClock, should_emit, sustainable_chunk


def test_clock_queues_back_to_back_and_restarts_after_running_dry():
    now = [10.0]
    clock = PlaybackClock(latency=0.1, now=lambda: now[0])
    assert clock.add(1.0) == 10.1
    assert clock.add(0.5) == 11.1  # queued behind the first chunk
    assert abs(clock.queued() - 1.5) < 1e-9
    now[0] = 20.0
    assert clock.queued() == 0.0
    assert clock.add(0.5) == 20.1  # after a gap, plays on arrival


def test_cost_model_tracks_observations():
    cost = CostModel()
    for _ in range(10):
        cost.observe_generation(8, 0.8)
        cost.observe_decode(100, 1.0)
    assert abs(cost.generate - 0.1) < 1e-3 and abs(cost.decode - 0.01) < 1e-3


FRAME = 2048 / 44100


def _cost(generate=0.02, decode=0.005):
    cost = CostModel()
    cost.generate, cost.decode = generate, decode
    return cost


def _queued(seconds):
    clock = PlaybackClock(latency=0.0, now=lambda: 0.0)
    clock.playhead = seconds
    return clock


def test_emits_when_waiting_would_starve_and_not_before():
    # Waiting costs 8 * 0.02 + 108 * 0.005 + 0.15 = 0.85 s of queued audio; 40 pending frames is sustainable.
    assert should_emit(40, 100, 8, FRAME, _queued(0.8), _cost())
    assert not should_emit(40, 100, 8, FRAME, _queued(0.9), _cost())


def test_never_emits_a_chunk_too_small_to_keep_up():
    # At 100 ready frames a chunk must outlast 1.5 * 0.5 s of decode with 26 ms of spare time per frame.
    assert sustainable_chunk(100, FRAME, _cost()) > 25
    assert not should_emit(20, 100, 8, FRAME, _queued(0.0), _cost())
    assert should_emit(30, 100, 8, FRAME, _queued(0.0), _cost())


def test_the_first_chunk_of_an_utterance_is_larger():
    fresh = _queued(0.0)
    fresh.speech_written = False
    assert not should_emit(10, 10, 8, FRAME, fresh, _cost(decode=0.0))
    assert should_emit(15, 15, 8, FRAME, fresh, _cost(decode=0.0))


def test_generation_slower_than_real_time_waits_for_one_final_decode():
    assert sustainable_chunk(10, FRAME, _cost(generate=FRAME * 2)) == float("inf")
