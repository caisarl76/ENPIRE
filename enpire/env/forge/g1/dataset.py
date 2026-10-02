"""Inspect LeRobot G1 handover evidence without replaying or modifying it."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def _joint_names() -> list[str]:
    legs = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    arms = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")
    hands = ("hand_index_0", "hand_index_1", "hand_middle_0", "hand_middle_1",
             "hand_thumb_0", "hand_thumb_1", "hand_thumb_2")
    return ([f"{side}_{joint}_joint" for side in ("left", "right") for joint in legs]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [f"{side}_{joint}_joint" for side in ("left", "right") for joint in arms + hands])


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inspect_dataset(root: Path) -> dict:
    """Validate the recorded observation/action contract and return descriptive evidence."""
    import numpy as np
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This inspector expects LeRobot v2.1 metadata")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be positive and finite")
    sizes = {"observation.state": 43, "action.wbc": 43, "observation.eef_state": 14,
             "action.motion_token": 64, "teleop.left_hand_joints": 7,
             "teleop.right_hand_joints": 7, "teleop.stream_mode": 1}
    for key, size in sizes.items():
        feature = info.get("features", {}).get(key, {})
        if feature.get("shape") != [size]:
            raise ValueError(f"{key} metadata must declare shape [{size}]")
        if key in ("observation.state", "action.wbc") and feature.get("names") != _joint_names():
            raise ValueError(f"{key} joint order does not match the G1 Dex3 contract")
    episodes = _json_lines(root / "meta/episodes.jsonl")
    if not episodes or len(episodes) != info.get("total_episodes"):
        raise ValueError("Episode count does not match metadata")
    if [ep.get("episode_index") for ep in episodes] != list(range(len(episodes))):
        raise ValueError("Episode indices must be unique, contiguous, and ordered")
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError("Dataset requires a data_path template")
    chunk_size = info.get("chunks_size", 1000)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunks_size must be a positive integer")
    counts = Counter()
    summaries = []
    checked_frames = 0
    seen_paths = set()
    for episode in episodes:
        index, length = episode["episode_index"], episode.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"Episode {index} must have a positive integer length")
        try:
            relative = template.format(episode_chunk=index // chunk_size, episode_index=index)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"Episode {index}: invalid data_path template") from error
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset data_path must remain inside its root")
        if path in seen_paths:
            raise ValueError(f"Episode {index}: episode Parquet path was reused")
        seen_paths.add(path)
        table = pq.read_table(path, columns=list(sizes) + ["episode_index", "frame_index"]).to_pydict()
        for key, expected in (("episode_index", np.full(length, index)),
                              ("frame_index", np.arange(length))):
            actual = np.asarray(table[key]).reshape(-1)
            if actual.shape != (length,) or not np.array_equal(actual, expected):
                raise ValueError(f"Episode {index}: {key} does not match metadata or frame sequence")
        arrays = {}
        for key, size in sizes.items():
            try:
                array = np.asarray(table[key], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Episode {index}: invalid {key}") from error
            if size == 1 and array.shape == (length,):
                array = array.reshape(length, 1)
            if array.shape != (length, size) or not np.isfinite(array).all():
                raise ValueError(f"Episode {index}: {key} requires finite shape ({length}, {size})")
            arrays[key] = array
        modes = arrays["teleop.stream_mode"].ravel()
        if not np.equal(modes, np.floor(modes)).all() or not np.isin(modes, [0, 1, 2, 3, 4, 5]).all():
            raise ValueError(f"Episode {index}: invalid SONIC stream mode")
        episode_counts = Counter(map(int, modes))
        counts.update(episode_counts)
        state = arrays["observation.state"]
        summary = {"episode_index": index, "frames": length,
                   "stream_mode_counts": {str(k): v for k, v in sorted(episode_counts.items())}}
        for name, start, stop in (("left_arm", 15, 22), ("left_hand", 22, 29),
                                  ("right_arm", 29, 36), ("right_hand", 36, 43)):
            summary[f"{name}_range_l2"] = float(np.linalg.norm(np.ptp(state[:, start:stop], axis=0)))
        summaries.append(summary)
        checked_frames += length
    if checked_frames != info.get("total_frames"):
        raise ValueError("Frame count does not match metadata")
    return {
        "schema_version": 1, "dataset": str(root), "codebase_version": "v2.1", "fps": fps,
        "checked_episodes": len(summaries), "checked_frames": checked_frames,
        "tasks": _json_lines(root / "meta/tasks.jsonl"),
        "joint_names": _joint_names(), "observation_state_size": 43,
        "stream_mode_counts": {str(k): v for k, v in sorted(counts.items())},
        "stream_mode_names": {"0": "OFF", "1": "POSE", "2": "PLANNER",
                              "3": "PLANNER_FROZEN_UPPER_BODY", "4": "POSE_PAUSE", "5": "PLANNER_VR_3PT"},
        "episodes": summaries, "physical_success_verified": False, "motion_enabled": False,
        "limitations": [
            "Vector validity and movement ranges do not label grasp or handover success",
            "43-value action.wbc includes legs and is not a manipulation command for SONIC",
            "Recorded VR orientation uses 18 rotation-6D values; SONIC command quaternion fields differ",
            "EEF frames, hand semantics, camera calibration, and timing require station validation",
        ],
    }
