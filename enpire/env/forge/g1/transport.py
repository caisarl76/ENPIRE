"""Read the existing SONIC camera and g1_debug protocols without control sockets."""

from __future__ import annotations

import base64
import math
import time


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _vector(data: dict, name: str, size: int) -> list[float]:
    values = data.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} joint/component values")
    return [_number(value, name) for value in values]


def _unpack(raw: bytes) -> dict:
    import msgpack

    value = msgpack.unpackb(raw, raw=False)
    if not isinstance(value, dict):
        raise ValueError("Expected a MessagePack object")
    return value


def decode_state(raw: bytes, *, hand: str) -> dict:
    """Return measured state; dataset-compatible state is available only for Dex3."""
    if hand not in ("none", "dex3", "inspire_ftp"):
        raise ValueError("Unknown hand type")
    if not raw.startswith(b"g1_debug"):
        raise ValueError("Expected the g1_debug topic prefix")
    data = _unpack(raw[len(b"g1_debug"):])
    index = data.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("State index must be a nonnegative integer")
    body = _vector(data, "body_q", 29)
    quat = _vector(data, "base_quat", 4)
    if abs(sum(x * x for x in quat) - 1.0) > 0.02:
        raise ValueError("base_quat must be a unit quaternion in wxyz order")
    result = {"index": index, "timestamp": _number(data.get("ros_timestamp"), "ros_timestamp"),
              "body_q": body, "root_orientation": quat, "hand": hand}
    if hand == "dex3":
        left = _vector(data, "left_hand_q", 7)
        right = _vector(data, "right_hand_q", 7)
        result.update(left_hand_q=left, right_hand_q=right,
                      state=body[:22] + left + body[22:] + right)
    # Firmware variants may add torque or visualization fields; no map-size assumptions.
    return result


def decode_camera(raw: bytes) -> dict:
    """Decode the composed-camera ego JPEG into an RGB uint8 image."""
    import cv2
    import numpy as np

    data = _unpack(raw)
    images, timestamps = data.get("images"), data.get("timestamps")
    if not isinstance(images, dict) or not isinstance(timestamps, dict):
        raise ValueError("Camera message requires images and timestamps objects")
    encoded = images.get("ego_view")
    if isinstance(encoded, str):
        encoded = base64.b64decode(encoded, validate=True)
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("ego_view must contain a nonempty JPEG or base64 JPEG")
    timestamp = _number(timestamps.get("ego_view"), "ego_view timestamp")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("ego_view JPEG did not decode")
    return {"timestamp": timestamp, "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "jpeg_bytes": len(encoded)}


class Subscriber:
    """A bounded, subscription-only transport; never creates a command publisher."""

    def __init__(self, endpoint: str, stream: str, *, hand: str = "none"):
        import zmq

        if stream not in ("camera", "state"):
            raise ValueError("Stream must be camera or state")
        self.stream, self.hand = stream, hand
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"g1_debug" if stream == "state" else b"")
        try:
            self._socket.connect(endpoint)
        except Exception:
            self.close()
            raise

    def read_many(self, samples: int, timeout: float) -> list[dict]:
        """Require advancing source samples within one total deadline."""
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError("At least two samples are required to establish advancement")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        results = []
        while len(results) < samples:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.stream}: received {len(results)}/{samples} advancing samples; source absent or stalled")
            if not self._socket.poll(max(1, math.ceil(remaining * 1000))):
                continue
            raw = self._socket.recv()
            value = decode_camera(raw) if self.stream == "camera" else decode_state(raw, hand=self.hand)
            if results:
                if value["timestamp"] <= results[-1]["timestamp"]:
                    continue
                if self.stream == "state" and value["index"] <= results[-1]["index"]:
                    continue
            value["received_monotonic"] = time.monotonic()
            results.append(value)
        return results

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
