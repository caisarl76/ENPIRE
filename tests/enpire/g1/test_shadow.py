"""Recorded qualification must use v4 and end sessions at mode boundaries."""

import json

import pytest

from enpire.cli import main


def recorded_fixture(tmp_path, episodes):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from enpire.env.forge.g1.dataset import _joint_names

    root = tmp_path / "recorded"
    (root / "meta").mkdir(parents=True)
    sizes = {"observation.state": 43, "action.wbc": 43, "observation.eef_state": 14,
             "action.motion_token": 64, "teleop.left_hand_joints": 7,
             "teleop.right_hand_joints": 7, "teleop.stream_mode": 1}
    features = {key: {"shape": [size]} for key, size in sizes.items()}
    for key in ("observation.state", "action.wbc"):
        features[key]["names"] = _joint_names()
    info = dict(codebase_version="v2.1", fps=50, features=features,
                total_episodes=len(episodes), total_frames=sum(map(len, episodes)),
                data_path="data/episode_{episode_index:06d}.parquet")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": i, "length": len(modes)}) for i, modes in enumerate(episodes)))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "bottle presentation"}))
    (root / "data").mkdir()
    for episode, modes in enumerate(episodes):
        values = {key: [[0.0] * size for _ in modes] for key, size in sizes.items()}
        values.update({"teleop.stream_mode": [[mode] for mode in modes],
                       "teleop.right_hand_joints": [[10, 11, 12, 13, 14, 15, 16]] * len(modes),
                       "episode_index": [episode] * len(modes), "frame_index": list(range(len(modes))),
                       "timestamp": [i / 50 for i in range(len(modes))]})
        pq.write_table(pa.table(values), root / f"data/episode_{episode:06d}.parquet")
    return root


def encoder_fixture(tmp_path, *, before="", forced_version=None):
    path = tmp_path / "encoder.py"
    version_expression = "version" if forced_version is None else str(forced_version)
    path.write_text(f'''import json
{before}
HEADER_SIZE = 1280
def pack_pose_message(fields, *, topic, version):
    header = {{"v": {version_expression}, "endian": "le", "count": 1,
              "fields": [{{"name": key, "dtype": "i64" if key == "frame_index" else "f32",
                          "shape": list(value.shape)}} for key, value in fields.items()]}}
    return (topic.encode() + json.dumps(header).encode().ljust(1280, b"\\0")
            + b"".join(value.tobytes() for value in fields.values()))
''')
    return path


def test_qualification_splits_pose_runs_and_preserves_native_hand_order(tmp_path):
    from enpire.env.forge.g1.shadow import qualify_dataset

    root = recorded_fixture(tmp_path, [[1, 1, 2, 1], [1]])
    report = qualify_dataset(root, encoder_fixture(tmp_path))
    assert report["pose_records_checked"] == 4
    assert report["non_pose_records_skipped"] == 1
    assert report["pose_sessions"] == 3
    assert report["protocol_version"] == 4
    assert report["packet_sizes_bytes"] == [1604]
    assert report["roundtrip_exact_after_float32_cast"] is True
    assert report["motion_enabled"] is False
    assert report["receiver_runtime_verified"] is False
    assert report["packets_saved_or_sent"] is False


def test_version_three_token_only_packets_fail_qualification(tmp_path):
    from enpire.env.forge.g1.shadow import qualify_dataset

    root = recorded_fixture(tmp_path, [[1]])
    with pytest.raises(ValueError, match="version 4"):
        qualify_dataset(root, encoder_fixture(tmp_path, forced_version=3))


def test_socket_attempt_cannot_be_hidden_by_the_encoder(tmp_path):
    from enpire.env.forge.g1.shadow import qualify_dataset

    root = recorded_fixture(tmp_path, [[1]])
    before = "import socket\ntry:\n    socket.socket()\nexcept OSError:\n    pass"
    with pytest.raises(ValueError, match="socket"):
        qualify_dataset(root, encoder_fixture(tmp_path, before=before))


def test_shadow_cli_preserves_existing_report_without_loading_encoder(tmp_path, capsys):
    output = tmp_path / "report.json"
    output.write_text("existing evidence")
    status = main(["g1", "shadow-adapter", "--dataset", str(tmp_path / "missing"),
                   "--encoder", str(tmp_path / "missing.py"), "--output", str(output)])
    assert status == 2
    assert output.read_text() == "existing evidence"
    assert json.loads(capsys.readouterr().out)["motion_enabled"] is False


def test_shadow_cli_writes_a_summary_without_packet_data(tmp_path, capsys):
    root = recorded_fixture(tmp_path, [[1, 1]])
    encoder = encoder_fixture(tmp_path)
    output = tmp_path / "qualification.json"
    assert main(["g1", "shadow-adapter", "--dataset", str(root), "--encoder", str(encoder),
                 "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["pose_records_checked"] == 2
    assert report["physical_success_verified"] is False
    assert "packets" not in report
    assert json.loads(capsys.readouterr().out)["motion_enabled"] is False
