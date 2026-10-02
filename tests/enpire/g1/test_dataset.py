"""Offline inspection must reject data that cannot match the G1 observation contract."""

import json

import numpy as np
import pytest

from enpire.cli import main
from enpire.env.forge.g1.dataset import inspect_dataset


def fixture_dataset(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Explicitly derive the documented LeRobot order independently of production code.
    legs = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in
            ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    arm = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
           "wrist_roll", "wrist_pitch", "wrist_yaw")
    hand = ("hand_index_0", "hand_index_1", "hand_middle_0", "hand_middle_1",
            "hand_thumb_0", "hand_thumb_1", "hand_thumb_2")
    names = legs + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    for side in ("left", "right"):
        names += [f"{side}_{joint}_joint" for joint in arm + hand]
    features = {"observation.state": {"shape": [43], "names": names},
                "action.wbc": {"shape": [43], "names": names},
                "observation.eef_state": {"shape": [14]},
                "action.motion_token": {"shape": [64]},
                "teleop.left_hand_joints": {"shape": [7]},
                "teleop.right_hand_joints": {"shape": [7]},
                "teleop.stream_mode": {"shape": [1]},
                "episode_index": {"shape": [1]}, "frame_index": {"shape": [1]}}
    info = {"codebase_version": "v2.1", "total_episodes": 1, "total_frames": 3, "fps": 50,
            "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test bottle handover"}) + "\n")
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    state = np.zeros((3, 43))
    state[1:, 29] = [1, 2]  # Right shoulder; left-hand slot 22 does not count as right arm.
    state[1:, 22] = [5, 10]
    values = {key: np.zeros((3, value["shape"][0])).tolist() for key, value in features.items()}
    values["observation.state"] = state.tolist()
    values["teleop.stream_mode"] = [1, 1, 2]  # Real Parquet stores scalar features as scalars.
    values["episode_index"] = [0, 0, 0]
    values["frame_index"] = [0, 1, 2]
    path = root / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(values), path)
    return info, values, path


def test_dataset_reports_motion_ranges_without_confusing_hand_and_arm(tmp_path):
    pytest.importorskip("pyarrow")
    fixture_dataset(tmp_path)
    report = inspect_dataset(tmp_path)
    assert report["checked_frames"] == 3
    assert report["stream_mode_counts"] == {"1": 2, "2": 1}
    episode = report["episodes"][0]
    assert episode["left_arm_range_l2"] == 0
    assert episode["right_arm_range_l2"] == 2
    assert episode["left_hand_range_l2"] == 10
    assert report["physical_success_verified"] is False


def test_incorrect_joint_order_cannot_pass_by_matching_only_vector_length(tmp_path):
    pytest.importorskip("pyarrow")
    info, _, _ = fixture_dataset(tmp_path)
    names = info["features"]["observation.state"]["names"]
    names[22], names[29] = names[29], names[22]
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint order"):
        inspect_dataset(tmp_path)


@pytest.mark.parametrize("damage", ["missing", "nan", "count", "shape", "bad_mode"])
def test_missing_or_corrupt_episode_never_reports_a_valid_dataset(tmp_path, damage):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    info, values, path = fixture_dataset(tmp_path)
    if damage == "missing":
        path.unlink()
    elif damage == "count":
        info["total_frames"] = 4
        (tmp_path / "meta/info.json").write_text(json.dumps(info))
    else:
        if damage == "nan":
            values["action.motion_token"][1][0] = float("nan")
        elif damage == "shape":
            values["teleop.right_hand_joints"][2] = [0.0] * 6
        else:
            values["teleop.stream_mode"][0] = 1.5
        pq.write_table(pa.table(values), path)
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_dataset(tmp_path)


def test_one_parquet_file_cannot_be_counted_as_two_episodes(tmp_path):
    pytest.importorskip("pyarrow")
    info, _, _ = fixture_dataset(tmp_path)
    info.update(total_episodes=2, total_frames=6,
                data_path="data/chunk-000/episode_000000.parquet")
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "meta/episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 3}\n{"episode_index": 1, "length": 3}\n')
    with pytest.raises(ValueError, match="reused"):
        inspect_dataset(tmp_path)


@pytest.mark.parametrize("column, values", [
    ("episode_index", [0, 1, 0]), ("frame_index", [0, 2, 2])])
def test_episode_identity_and_frame_sequence_must_match_metadata(tmp_path, column, values):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, data, path = fixture_dataset(tmp_path)
    data[column] = values
    pq.write_table(pa.table(data), path)
    with pytest.raises(ValueError, match=column):
        inspect_dataset(tmp_path)


def test_bad_path_template_returns_actionable_cli_failure(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    info, _, _ = fixture_dataset(tmp_path)
    info["data_path"] = "data/{unknown_chunk}/episode_{episode_index:06d}.parquet"
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    assert main(["g1", "inspect-dataset", "--dataset", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "data_path template" in report["error"]
    assert report["motion_enabled"] is False
