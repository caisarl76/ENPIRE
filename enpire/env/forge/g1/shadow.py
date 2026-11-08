"""Qualify recorded SONIC v4 outputs in memory, with socket creation blocked."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from unittest.mock import patch

from .adapter import MemoryBus, OfflineAdapter, PoseTarget
from .dataset import inspect_dataset


def _load_encoder(path):
    spec = importlib.util.spec_from_file_location("enpire_offline_sonic_encoder", path)
    if spec is None or spec.loader is None:
        raise ValueError("Encoder must be a local Python source file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "HEADER_SIZE", None) != 1280 or not callable(getattr(module, "pack_pose_message", None)):
        raise ValueError("Expected the SONIC 1280-byte-header pack_pose_message encoder")
    return module.pack_pose_message


def _check_packet(packet, target):
    """Check the boundary bytes independently of adapter field construction."""
    import numpy as np

    if len(packet) < 1284 or not packet.startswith(b"pose"):
        raise ValueError("Malformed pose packet")
    header = json.loads(packet[4:1284].rstrip(b"\0"))
    if header.get("v") != 4:
        raise ValueError("Token-only packets require protocol version 4")
    if header.get("endian") != "le" or header.get("count") != 1:
        raise ValueError("Expected one little-endian pose record")
    expected = {"token_state": ("f32", [1, 64], np.asarray([target.token], dtype="<f4")),
                "frame_index": ("i64", [1], np.asarray([target.frame_index], dtype="<i8")),
                "left_hand_joints": ("f32", [1, 7], np.asarray([target.left_hand], dtype="<f4")),
                "right_hand_joints": ("f32", [1, 7], np.asarray([target.right_hand], dtype="<f4"))}
    fields = header.get("fields")
    if not isinstance(fields, list) or len(fields) != 4:
        raise ValueError("Expected the four token/hand/frame fields")
    cursor, seen = 1284, set()
    for field in fields:
        name = field.get("name")
        if name not in expected or name in seen:
            raise ValueError("Unexpected or repeated pose field")
        dtype, shape, values = expected[name]
        if field.get("dtype") != dtype or field.get("shape") != shape:
            raise ValueError(f"Incorrect wire shape or dtype for {name}")
        restored = np.frombuffer(packet, dtype=values.dtype, count=values.size,
                                 offset=cursor).reshape(shape)
        if not np.array_equal(restored, values):
            raise ValueError(f"Wire round-trip changed {name}")
        cursor += values.nbytes
        seen.add(name)
    if cursor != len(packet):
        raise ValueError("Pose packet has trailing or missing bytes")


class _RecordedClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def qualify_dataset(dataset: Path, encoder_path: Path) -> dict:
    """Reference actor qualification; no autonomous policy or receiver execution."""
    import numpy as np
    import pyarrow.parquet as pq

    root, source = Path(dataset).resolve(), Path(encoder_path).resolve()
    attempts = 0

    def reject_socket(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise PermissionError("Offline qualification prohibits socket creation")

    checked = skipped = sessions = 0
    packet_sizes = set()
    clock = _RecordedClock()
    # Imports above finish before the guard. The explicit upstream module is a
    # pure byte builder; the guard also catches attempted standard-library IO.
    with patch.object(socket, "socket", reject_socket):
        inspection = inspect_dataset(root)
        info = json.loads((root / "meta/info.json").read_text())
        encoder = _load_encoder(source)
        period = 1.0 / inspection["fps"]
        for episode in inspection["episodes"]:
            path = root / info["data_path"].format(
                episode_index=episode["episode_index"],
                episode_chunk=episode["episode_index"] // info.get("chunks_size", 1000))
            names = ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints",
                     "teleop.stream_mode", "frame_index", "timestamp"]
            table = pq.read_table(path, columns=names).to_pydict()
            stamps = np.asarray(table["timestamp"], dtype=float).reshape(-1)
            if (stamps.shape != (episode["frames"],) or not np.isfinite(stamps).all()
                    or np.any(stamps < 0) or np.any(np.diff(stamps) <= 0)):
                raise ValueError("Recorded timestamps must be finite, nonnegative, and advancing")
            adapter = None
            sample_index = 1
            try:
                for row, stamp in enumerate(stamps):
                    clock.now += float(stamp - stamps[row-1]) if row else period
                    mode = int(np.asarray(table["teleop.stream_mode"][row]).reshape(-1)[0])
                    if mode != 1:
                        skipped += 1
                        if adapter is not None:
                            adapter.stop()
                            adapter = None
                        continue
                    if adapter is None:
                        bus = MemoryBus()
                        adapter = OfflineAdapter(bus, encoder, clock=clock, owner="recorded_reference")
                        adapter.observe_state(0, clock.now, clock.now)
                        adapter.observe_camera(clock.now, clock.now)
                        clock.now += period  # Explicit synthetic startup advancement.
                        sample_index = 1
                        adapter.observe_state(sample_index, clock.now, clock.now)
                        adapter.observe_camera(clock.now, clock.now)
                        if not adapter.start():
                            raise ValueError(f"Reference session refused start: {adapter.report()['reason']}")
                        sessions += 1
                    else:
                        if not adapter.tick():
                            raise ValueError(f"Reference watchdog fault: {adapter.report()['reason']}")
                        sample_index += 1
                        adapter.observe_state(sample_index, clock.now, clock.now)
                        adapter.observe_camera(clock.now, clock.now)
                    target = PoseTarget(
                        token=tuple(table["action.motion_token"][row]),
                        left_hand=tuple(table["teleop.left_hand_joints"][row]),
                        right_hand=tuple(table["teleop.right_hand_joints"][row]),
                        hand_order="dex3_motor", frame_index=int(table["frame_index"][row]),
                        created_monotonic=clock.now, input_received_monotonic=clock.now)
                    if not adapter.submit(target):
                        raise ValueError(f"Reference output rejected: {adapter.report()['reason']}")
                    packet = bus.packets[-1]
                    _check_packet(packet, target)
                    packet_sizes.add(len(packet))
                    checked += 1
            finally:
                if adapter is not None:
                    adapter.stop()
        if attempts:
            raise ValueError("The encoder attempted socket creation during offline qualification")
    if checked + skipped != inspection["checked_frames"] or not checked:
        raise ValueError("Qualification requires matching counts and at least one POSE record")
    return {"schema_version": 1, "dataset": str(root), "encoder_source": str(source),
            "encoder_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "checked_episodes": inspection["checked_episodes"], "pose_records_checked": checked,
            "non_pose_records_skipped": skipped, "pose_sessions": sessions,
            "protocol_version": 4, "header_size_bytes": 1280,
            "packet_sizes_bytes": sorted(packet_sizes), "roundtrip_exact_after_float32_cast": True,
            "standard_library_socket_creation_prohibited": True, "socket_creation_attempts": attempts,
            "packets_saved_or_sent": False, "receipt_clock": "synthetic, recorded inter-frame cadence",
            "source_advancement": "synthetic state/camera samples; no live streams",
            "controller_choice": "staged task supervisor with a SONIC motion-token actor",
            "policy_checkpoint_selected": False, "receiver_runtime_verified": False,
            "receiver_fallback_verified": False, "physical_success_verified": False,
            "image_state_alignment_verified": False, "motion_enabled": False,
            "limitations": [
                "Reference actor uses saved POSE tokens; autonomous bottle policy is not implemented",
                "Memory leases model local ownership; real publisher arbitration remains unverified",
                "Stopping output does not clear receiver-held tokens or establish a safe robot fallback",
                "Whole-body token behavior, hand operating limits, ramps, and reset need characterization",
            ]}
