"""A motion-token adapter must refuse stale, competing, or malformed outputs."""

import threading
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def setup_adapter(**kwargs):
    from enpire.env.forge.g1.adapter import MemoryBus, OfflineAdapter, PoseTarget

    clock = Clock()
    bus = kwargs.pop("bus", MemoryBus())
    captured = []

    def encoder(fields, *, topic, version):
        captured.append((fields, topic, version))
        return b"pose-test-only"

    adapter = OfflineAdapter(bus, kwargs.pop("encoder", encoder), clock=clock,
                             owner=kwargs.pop("owner", "task"), **kwargs)
    for index, stamp in ((0, 0.0), (1, 0.02)):
        clock.now = stamp
        adapter.observe_state(index, stamp, stamp)
        adapter.observe_camera(stamp, stamp)
    target = PoseTarget(token=tuple(range(64)), left_hand=(0, 1, 2, 3, 4, 5, 6),
                        right_hand=(10, 11, 12, 13, 14, 15, 16), hand_order="dataset",
                        frame_index=0, created_monotonic=0.02,
                        input_received_monotonic=0.02)
    return adapter, bus, clock, target, captured


def test_dataset_hand_targets_are_reordered_and_token_only_protocol_is_v4():
    adapter, bus, _, target, captured = setup_adapter()
    assert adapter.start()
    assert adapter.submit(target)
    fields, topic, version = captured[0]
    assert topic == "pose" and version == 4
    assert set(fields) == {"token_state", "frame_index", "left_hand_joints", "right_hand_joints"}
    assert fields["token_state"].shape == (1, 64)
    assert fields["token_state"].dtype == np.float32
    assert fields["frame_index"].dtype == np.int64
    assert fields["frame_index"].tolist() == [0]
    assert fields["left_hand_joints"].tolist() == [[4, 5, 6, 0, 1, 2, 3]]
    assert fields["right_hand_joints"].tolist() == [[14, 15, 16, 10, 11, 12, 13]]
    assert len(bus.packets) == 1
    assert adapter.report()["motion_enabled"] is False


def test_native_hand_targets_are_preserved():
    adapter, _, _, target, captured = setup_adapter()
    assert adapter.start()
    assert adapter.submit(replace(target, hand_order="dex3_motor"))
    assert captured[0][0]["right_hand_joints"].tolist() == [[10, 11, 12, 13, 14, 15, 16]]


def test_start_requires_two_advancing_camera_and_state_samples():
    from enpire.env.forge.g1.adapter import MemoryBus, OfflineAdapter

    clock = Clock()
    adapter = OfflineAdapter(MemoryBus(), lambda *_: b"unused", clock=clock, owner="task")
    adapter.observe_state(0, 0, 0)
    adapter.observe_camera(0, 0)
    assert not adapter.start()
    assert adapter.report()["state"] == "disarmed"


@pytest.mark.parametrize("stream", ["state", "camera"])
def test_duplicate_samples_do_not_refresh_freshness(stream):
    adapter, bus, clock, target, _ = setup_adapter()
    assert adapter.start()
    clock.now = 0.25
    if stream == "state":
        adapter.observe_state(1, 0.02, clock.now)
        adapter.observe_camera(0.25, clock.now)
    else:
        adapter.observe_camera(0.02, clock.now)
        adapter.observe_state(2, 0.25, clock.now)
    assert not adapter.submit(replace(target, created_monotonic=clock.now,
                                      input_received_monotonic=clock.now))
    assert adapter.report()["reason"] == f"{stream}_stale"
    assert not bus.packets


def test_recent_actor_output_cannot_hide_stale_policy_input():
    adapter, bus, clock, target, _ = setup_adapter()
    assert adapter.start()
    clock.now = 0.25
    adapter.observe_state(2, 0.25, clock.now)
    adapter.observe_camera(0.25, clock.now)
    assert not adapter.submit(replace(target, created_monotonic=0.25, input_received_monotonic=0.0))
    assert adapter.report()["reason"] == "policy_input_stale"
    assert not bus.packets


def test_idle_watchdog_stops_output_and_requires_a_new_session():
    adapter, bus, clock, target, _ = setup_adapter()
    assert adapter.start()
    assert adapter.submit(target)
    clock.now = 0.121
    adapter.observe_state(2, 0.121, clock.now)
    adapter.observe_camera(0.121, clock.now)
    assert not adapter.tick()
    assert adapter.report()["reason"] == "actor_stale"
    assert not adapter.start()
    assert not adapter.submit(replace(target, frame_index=1, created_monotonic=clock.now,
                                      input_received_monotonic=clock.now))
    assert len(bus.packets) == 1


def test_explicit_stop_does_not_emit_a_release_or_zero_token():
    adapter, bus, _, target, _ = setup_adapter()
    assert adapter.start()
    assert adapter.submit(target)
    adapter.stop()
    assert adapter.report()["state"] == "stopped"
    assert not adapter.start()
    assert not adapter.submit(replace(target, frame_index=1))
    assert len(bus.packets) == 1
    assert adapter.report()["receiver_fallback_verified"] is False


def test_competing_writer_cannot_acquire_and_teleoperation_revokes_task_writer():
    adapter, bus, clock, target, _ = setup_adapter()
    assert adapter.start()
    assert bus.acquire("other", clock.now, 1) is None
    bus.set_teleop_active(True)
    assert not adapter.submit(target)
    assert adapter.report()["reason"] == "ownership_lost"
    assert not bus.packets
    assert bus.acquire("other", clock.now, 1) is None


def test_expired_lease_cannot_commit_or_release_a_new_owners_lease():
    from enpire.env.forge.g1.adapter import MemoryBus

    bus = MemoryBus()
    first = bus.acquire("first", 0, 0.1)
    second = bus.acquire("second", 0.1, 0.1)
    assert second is not None
    with pytest.raises(ValueError, match="ownership"):
        bus.publish(first, b"old", 0.1, 0.1)
    bus.release(first)
    bus.publish(second, b"new", 0.1, 0.1)
    assert bus.packets == (b"new",)


@pytest.mark.parametrize("damage,reason", [
    ({"token": (0,) * 63}, "invalid_target"),
    ({"right_hand": (0,) * 6}, "invalid_target"),
    ({"token": (float("nan"),) * 64}, "invalid_target"),
    ({"token": (1e40,) * 64}, "invalid_target"),
    ({"hand_order": "guess"}, "invalid_target"),
    ({"frame_index": True}, "invalid_target"),
    ({"frame_index": 2**63}, "invalid_target"),
    ({"stream_mode": 2}, "invalid_target"),
    ({"created_monotonic": 1.0}, "invalid_target"),
    ({"input_received_monotonic": 0.03}, "invalid_target"),
])
def test_invalid_actor_outputs_latch_fault_before_encoding(damage, reason):
    adapter, bus, _, target, captured = setup_adapter()
    assert adapter.start()
    assert not adapter.submit(replace(target, **damage))
    assert adapter.report()["reason"] == reason
    assert not bus.packets and not captured


def test_repeated_target_frame_is_rejected_without_retransmitting():
    adapter, bus, _, target, _ = setup_adapter()
    assert adapter.start() and adapter.submit(target)
    assert not adapter.submit(target)
    assert adapter.report()["reason"] == "frame_not_advancing"
    assert len(bus.packets) == 1


@pytest.mark.parametrize("stream", ["state", "camera"])
def test_source_reset_latches_fault(stream):
    adapter, _, clock, _, _ = setup_adapter()
    assert adapter.start()
    clock.now = 0.03
    if stream == "state":
        adapter.observe_state(0, 0.0, clock.now)
    else:
        adapter.observe_camera(0.0, clock.now)
    assert adapter.report()["reason"] == f"{stream}_reset"
    assert not adapter.tick()


def test_future_receipt_time_and_clock_regression_fail_closed():
    adapter, bus, clock, _, _ = setup_adapter()
    assert adapter.start()
    adapter.observe_camera(0.03, 1)
    assert adapter.report()["reason"] == "invalid_camera"
    assert not bus.packets
    adapter, bus, clock, _, _ = setup_adapter()
    assert adapter.start()
    clock.now = 0.01
    assert not adapter.tick()
    assert adapter.report()["reason"] == "invalid_clock"


def test_encoder_failure_stops_session_without_partial_output():
    def broken(*args, **kwargs):
        raise RuntimeError("encoder failed")

    adapter, bus, _, target, _ = setup_adapter(encoder=broken)
    assert adapter.start()
    assert not adapter.submit(target)
    assert adapter.report()["reason"] == "encoder_failed"
    assert not bus.packets


def test_encoder_delay_rechecks_freshness_before_committing():
    adapter, bus, clock, target, _ = setup_adapter()

    def slow(*args, **kwargs):
        clock.now = 0.25
        return b"pose-test-only"

    adapter.encoder = slow
    assert adapter.start()
    assert not adapter.submit(target)
    assert adapter.report()["reason"] == "state_stale"
    assert not bus.packets


def test_takeover_during_encoding_rechecks_ownership_before_committing():
    adapter, bus, _, target, _ = setup_adapter()

    def takeover(*args, **kwargs):
        bus.set_teleop_active(True)
        return b"pose-test-only"

    adapter.encoder = takeover
    assert adapter.start()
    assert not adapter.submit(target)
    assert adapter.report()["reason"] == "ownership_lost"
    assert not bus.packets


def test_overlapping_submissions_cancel_inflight_work_instead_of_reordering_frames():
    adapter, bus, _, target, _ = setup_adapter()
    entered, resume = threading.Event(), threading.Event()
    outcomes = []

    def blocking(fields, **kwargs):
        if int(fields["frame_index"][0]) == 0:
            entered.set()
            assert resume.wait(2)
        return b"pose-test-only"

    adapter.encoder = blocking
    assert adapter.start()
    worker = threading.Thread(target=lambda: outcomes.append(adapter.submit(target)))
    worker.start()
    try:
        assert entered.wait(2)
        assert not adapter.submit(replace(target, frame_index=1))
    finally:
        resume.set()
        worker.join(2)
    assert not worker.is_alive()
    assert outcomes == [False]
    assert not bus.packets
    assert adapter.report()["reason"] == "concurrent_output"


def test_stop_during_encoding_cancels_work_and_keeps_a_new_owners_lease():
    adapter, bus, clock, target, _ = setup_adapter()
    entered, resume = threading.Event(), threading.Event()
    outcomes = []

    def blocking(*args, **kwargs):
        entered.set()
        assert resume.wait(2)
        return b"pose-test-only"

    adapter.encoder = blocking
    assert adapter.start()
    worker = threading.Thread(target=lambda: outcomes.append(adapter.submit(target)))
    worker.start()
    try:
        assert entered.wait(2)
        adapter.stop()
        new_lease = bus.acquire("other", clock.now, 1)
        assert new_lease is not None
    finally:
        resume.set()
        worker.join(2)
    assert outcomes == [False]
    assert not bus.packets
    assert adapter.report()["state"] == "stopped"
    assert bus.valid(new_lease, clock.now)


def test_observed_lease_expiration_cannot_be_reversed_by_a_decreasing_clock():
    from enpire.env.forge.g1.adapter import MemoryBus

    bus = MemoryBus()
    lease = bus.acquire("task", 0, 0.1)
    assert not bus.valid(lease, 0.2)
    with pytest.raises(ValueError):
        bus.publish(lease, b"expired", 0.05, 0.1)
    assert not bus.packets


def test_clock_failure_latches_fault_even_if_the_clock_recovers():
    adapter, bus, clock, target, _ = setup_adapter()
    assert adapter.start()

    def broken():
        raise RuntimeError("Clock provider failed")

    adapter.clock = broken
    assert not adapter.tick()
    assert adapter.report()["reason"] == "invalid_clock"
    adapter.clock = clock
    assert not adapter.submit(target)
    assert not adapter.start()
    assert not bus.packets
