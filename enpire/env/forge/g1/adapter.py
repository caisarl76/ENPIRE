"""Socket-free qualification of a SONIC v4 motion-token policy boundary."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from functools import wraps
from numbers import Integral, Real


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return float(value)


def _time(value):
    value = _finite(value)
    if value < 0:
        raise ValueError("Time must be nonnegative")
    return value


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


@dataclass(frozen=True)
class Limits:
    """Software qualification parameters; not validated physical limits."""

    max_state_age: float = 0.1
    max_camera_age: float = 0.2
    max_target_age: float = 0.1
    max_input_age: float = 0.2
    lease_duration: float = 0.3

    def __post_init__(self):
        if any(_finite(value) <= 0 for value in asdict(self).values()):
            raise ValueError("Adapter limits must be positive and finite")


@dataclass(frozen=True)
class PoseTarget:
    token: tuple[float, ...]
    left_hand: tuple[float, ...]
    right_hand: tuple[float, ...]
    hand_order: str
    frame_index: int
    created_monotonic: float
    input_received_monotonic: float
    stream_mode: int = 1


@dataclass(frozen=True)
class Lease:
    owner: str
    generation: int
    expires_at: float


class MemoryBus:
    """Exclusive writer model with atomic in-memory commits; no transport."""

    def __init__(self):
        self._lease = None
        self._generation = 0
        self._teleop = False
        self._packets = []
        self._lock = threading.RLock()
        self._last_time = 0.0

    def _advance(self, now):
        if now < self._last_time:
            raise ValueError("Ownership clock regressed")
        self._last_time = now
        if self._lease is not None and now >= self._lease.expires_at:
            self._lease = None

    @property
    def packets(self):
        with self._lock:
            return tuple(self._packets)

    def acquire(self, owner, now, duration):
        now, duration = _time(now), _finite(duration)
        if not isinstance(owner, str) or not owner.strip() or duration <= 0:
            raise ValueError("Ownership requires an owner and positive lease duration")
        with self._lock:
            self._advance(now)
            if self._teleop or (self._lease is not None and now < self._lease.expires_at):
                return None
            self._generation += 1
            self._lease = Lease(owner, self._generation, _time(now + duration))
            return self._lease

    def valid(self, lease, now):
        now = _time(now)
        with self._lock:
            self._advance(now)
            return (not self._teleop and lease is not None and self._lease == lease
                    and now < lease.expires_at)

    def publish(self, lease, packet, now, duration):
        """Commit and renew atomically; previous lease handles become invalid."""
        now, duration = _time(now), _finite(duration)
        if not isinstance(packet, bytes) or not packet or duration <= 0:
            raise ValueError("Expected packet bytes and a positive lease duration")
        with self._lock:
            if not self.valid(lease, now):
                raise ValueError("Command ownership was lost")
            self._generation += 1
            self._lease = Lease(lease.owner, self._generation, _time(now + duration))
            self._packets.append(packet)
            return self._lease

    def release(self, lease):
        with self._lock:
            if lease is not None and self._lease == lease:
                self._lease = None

    def set_teleop_active(self, active):
        if not isinstance(active, bool):
            raise ValueError("Teleoperation status must be boolean")
        with self._lock:
            self._teleop = active
            if active:
                self._lease = None


class OfflineAdapter:
    """Gate an injected encoder into MemoryBus; never start or stop hardware."""

    def __init__(self, bus, encoder, *, clock, owner, limits=None):
        if not isinstance(bus, MemoryBus) or not callable(encoder) or not callable(clock):
            raise ValueError("Offline adapter requires MemoryBus, encoder, and clock")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Adapter owner must be nonempty")
        self.bus, self.encoder, self.clock, self.owner = bus, encoder, clock, owner
        self.limits = limits if limits is not None else Limits()
        if not isinstance(self.limits, Limits):
            raise ValueError("Expected adapter Limits")
        self._last_clock = _time(clock())
        self._state, self._reason = "disarmed", "awaiting_sources"
        self._lease = None
        self._samples = {"state": None, "camera": None}
        self._counts = {"state": 0, "camera": 0}
        self._last_frame = -1
        self._last_actor_time = None
        self._emitted = 0
        self._lock = threading.RLock()
        self._submit_lock = threading.Lock()

    @_locked
    def _fault(self, reason):
        if self._state not in ("fault", "stopped"):
            self.bus.release(self._lease)
            self._lease = None
            self._state, self._reason = "fault", reason
        return False

    def _now(self):
        try:
            now = _time(self.clock())
            if now < self._last_clock:
                raise ValueError("Clock regressed")
            self._last_clock = now
            return now
        except Exception:
            self._fault("invalid_clock")
            return None

    @_locked
    def _observe(self, stream, index, timestamp, received):
        if self._state in ("fault", "stopped"):
            return False
        now = self._now()
        if now is None:
            return False
        try:
            timestamp, received = _time(timestamp), _time(received)
            if received > now or isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise ValueError("Invalid source identity or receipt time")
        except (TypeError, ValueError, OverflowError):
            return self._fault(f"invalid_{stream}")
        previous = self._samples[stream]
        if previous is not None:
            old_index, old_timestamp, old_received = previous
            if index < old_index or timestamp < old_timestamp or received < old_received:
                return self._fault(f"{stream}_reset")
            if timestamp == old_timestamp or (stream == "state" and index == old_index):
                return False  # Repeated source data must not refresh liveness.
        self._samples[stream] = (index, timestamp, received)
        self._counts[stream] += 1
        return True

    def observe_state(self, index, timestamp, received_monotonic):
        return self._observe("state", index, timestamp, received_monotonic)

    def observe_camera(self, timestamp, received_monotonic):
        return self._observe("camera", 0, timestamp, received_monotonic)

    def _freshness(self, now):
        for stream, maximum in (("state", self.limits.max_state_age),
                                ("camera", self.limits.max_camera_age)):
            if self._samples[stream] is None or now - self._samples[stream][2] > maximum:
                return f"{stream}_stale"
        return None

    @_locked
    def start(self):
        if self._state != "disarmed":
            return False
        now = self._now()
        if now is None:
            return False
        if min(self._counts.values()) < 2:
            self._reason = "awaiting_advancing_sources"
            return False
        reason = self._freshness(now)
        if reason:
            self._reason = reason
            return False
        try:
            self._lease = self.bus.acquire(self.owner, now, self.limits.lease_duration)
        except ValueError:
            return self._fault("invalid_clock")
        if self._lease is None:
            self._reason = "ownership_busy"
            return False
        self._state, self._reason = "active", "ready"
        self._last_actor_time = now
        return True

    def _check(self, now):
        try:
            owned = self.bus.valid(self._lease, now)
        except ValueError:
            owned = False
        if not owned:
            return self._fault("ownership_lost")
        reason = self._freshness(now)
        return self._fault(reason) if reason else True

    @_locked
    def tick(self):
        """Runner watchdog: call even when the actor produces no output."""
        if self._state != "active":
            return False
        now = self._now()
        if now is None or not self._check(now):
            return False
        if now - self._last_actor_time > self.limits.max_target_age:
            return self._fault("actor_stale")
        return True

    def _fields(self, target, now):
        import numpy as np

        if (not isinstance(target, PoseTarget) or isinstance(target.frame_index, bool)
                or not isinstance(target.frame_index, Integral) or not 0 <= target.frame_index < 2**63
                or isinstance(target.stream_mode, bool) or target.stream_mode != 1
                or target.hand_order not in ("dataset", "dex3_motor")):
            raise ValueError("Invalid target identity, mode, or hand order")
        created, observed = _time(target.created_monotonic), _time(target.input_received_monotonic)
        if observed > created or created > now:
            raise ValueError("Target times must be ordered in the local clock domain")

        def vector(values, size):
            if not isinstance(values, (list, tuple)) or len(values) != size:
                raise ValueError("Incorrect target shape")
            result = [_finite(v) for v in values]
            if any(abs(v) > float(np.finfo(np.float32).max) for v in result):
                raise ValueError("Target cannot be represented as float32")
            return result

        token = vector(target.token, 64)
        left, right = vector(target.left_hand, 7), vector(target.right_hand, 7)
        if target.hand_order == "dataset":
            left, right = left[4:] + left[:4], right[4:] + right[:4]
        return {"token_state": np.asarray([token], dtype=np.float32),
                "frame_index": np.asarray([target.frame_index], dtype=np.int64),
                "left_hand_joints": np.asarray([left], dtype=np.float32),
                "right_hand_joints": np.asarray([right], dtype=np.float32)}

    def submit(self, target):
        if not self._submit_lock.acquire(blocking=False):
            return self._fault("concurrent_output")
        try:
            return self._submit(target)
        finally:
            self._submit_lock.release()

    def _submit(self, target):
        with self._lock:
            if self._state != "active":
                return False
            now = self._now()
            if now is None or not self._check(now):
                return False
            try:
                fields = self._fields(target, now)
            except (TypeError, ValueError, OverflowError):
                return self._fault("invalid_target")
            if target.frame_index <= self._last_frame:
                return self._fault("frame_not_advancing")
            if now - target.input_received_monotonic > self.limits.max_input_age:
                return self._fault("policy_input_stale")
            if now - target.created_monotonic > self.limits.max_target_age:
                return self._fault("actor_stale")
            lease = self._lease
        # Do not hold the lifecycle lock across external work: stop and the
        # watchdog must be able to cancel a blocked encoder immediately.
        try:
            packet = self.encoder(fields, topic="pose", version=4)
            if not isinstance(packet, bytes) or not packet:
                raise ValueError("Encoder must return packet bytes")
        except Exception:
            return self._fault("encoder_failed")
        with self._lock:
            if self._state != "active":
                return False
            now = self._now()
            if now is None or not self._check(now):
                return False
            if self._lease != lease:
                return self._fault("ownership_lost")
            if target.frame_index <= self._last_frame:
                return self._fault("frame_not_advancing")
            if now - target.input_received_monotonic > self.limits.max_input_age:
                return self._fault("policy_input_stale")
            if now - target.created_monotonic > self.limits.max_target_age:
                return self._fault("actor_stale")
            try:
                self._lease = self.bus.publish(lease, packet, now, self.limits.lease_duration)
            except ValueError:
                return self._fault("ownership_lost")
            self._last_frame, self._last_actor_time = target.frame_index, target.created_monotonic
            self._emitted += 1
            return True

    @_locked
    def stop(self):
        if self._state not in ("fault", "stopped"):
            self.bus.release(self._lease)
            self._lease = None
            self._state, self._reason = "stopped", "explicit_stop"

    @_locked
    def report(self):
        return {"state": self._state, "reason": self._reason, "packets_in_memory": self._emitted,
                "software_limits": asdict(self.limits), "ownership_model": "in_memory_leases_only",
                "motion_enabled": False, "physical_success_verified": False,
                "receiver_fallback_verified": False, "image_state_alignment_verified": False}
