"""Catch protocol, joint-order, liveness, and readiness regressions."""

import base64
import builtins
import json
import math
import threading

import msgpack
import numpy as np
import pytest

from enpire.cli import main
from enpire.env.forge.g1.station import load_station, preflight
from enpire.env.forge.g1.transport import Subscriber, decode_camera, decode_state


def telemetry(**updates):
    data = {
        "index": 1,
        "ros_timestamp": 100.0,
        "body_q": list(range(29)),
        "left_hand_q": list(range(100, 107)),
        "right_hand_q": list(range(200, 207)),
        "base_quat": [1.0, 0.0, 0.0, 0.0],
    }
    data.update(updates)
    return b"g1_debug" + msgpack.packb(data, use_bin_type=True)


def test_dex3_state_maps_thumb_first_telemetry_to_dataset_joint_order():
    obs = decode_state(telemetry(motor_torque=[0.0] * 29), hand="dex3")
    # SONIC forwards the Dex3 motor order: thumb0/1/2, index0/1, middle0/1.
    # LeRobot stores index0/1, middle0/1, thumb0/1/2 after each arm.
    assert obs["state"][20:32] == [20, 21, 103, 104, 105, 106, 100, 101, 102, 22, 23, 24]
    assert obs["state"][-7:] == [203, 204, 205, 206, 200, 201, 202]
    assert obs["left_hand_q"] == [100, 101, 102, 103, 104, 105, 106]
    assert obs["right_hand_q"] == [200, 201, 202, 203, 204, 205, 206]
    assert len(obs["state"]) == 43


@pytest.mark.parametrize("hand", ["none", "inspire_ftp"])
def test_other_hands_do_not_create_a_dex3_observation(hand):
    obs = decode_state(telemetry(), hand=hand)
    assert "state" not in obs
    assert len(obs["body_q"]) == 29


@pytest.mark.parametrize("updates", [
    {"body_q": [0.0] * 28},
    {"left_hand_q": [0.0] * 6},
    {"right_hand_q": [math.nan] * 7},
    {"base_quat": [0.0] * 4},
    {"index": -1},
    {"ros_timestamp": math.inf},
])
def test_invalid_telemetry_cannot_be_used_as_an_observation(updates):
    with pytest.raises(ValueError):
        decode_state(telemetry(**updates), hand="dex3")


def test_state_rejects_wrong_topic():
    with pytest.raises(ValueError):
        decode_state(b"robot_config" + msgpack.packb({}), hand="dex3")


def test_camera_decodes_jpeg_to_rgb_and_rejects_missing_timestamp():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # OpenCV BGR red.
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    message = {"images": {"ego_view": base64.b64encode(jpeg).decode()},
               "timestamps": {"ego_view": 123.0}}
    decoded = decode_camera(msgpack.packb(message))
    assert decoded["image"].shape == (4, 6, 3)
    assert decoded["image"][0, 0, 0] > 240
    assert decoded["image"][0, 0, 2] < 10
    message["timestamps"] = {}
    with pytest.raises(ValueError):
        decode_camera(msgpack.packb(message))


def test_stalled_source_does_not_pass_even_while_messages_arrive():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def send():
        while not stop.wait(0.01):
            publisher.send(telemetry())

    thread = threading.Thread(target=send)
    thread.start()
    try:
        with Subscriber(f"tcp://127.0.0.1:{port}", "state", hand="dex3") as reader:
            with pytest.raises(TimeoutError):
                reader.read_many(2, 0.3)
    finally:
        stop.set()
        thread.join()
        publisher.close(linger=0)
        context.term()


def test_live_state_subscription_returns_advancing_source_samples():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def send():
        index = 1
        while not stop.wait(0.01):
            publisher.send(telemetry(index=index, ros_timestamp=100 + index / 50))
            index += 1

    thread = threading.Thread(target=send)
    thread.start()
    try:
        with Subscriber(f"tcp://127.0.0.1:{port}", "state", hand="dex3") as reader:
            samples = reader.read_many(3, 2.0)
        assert len(samples) == 3
        assert all(a["index"] < b["index"] for a, b in zip(samples, samples[1:]))
    finally:
        stop.set()
        thread.join()
        publisher.close(linger=0)
        context.term()


def test_station_cli_generates_external_profile_and_never_grants_motion(tmp_path, capsys):
    path = tmp_path / "station.json"
    args = ["g1", "init-station", "--output", str(path), "--robot-id", "test-g1",
            "--hand", "dex3", "--camera-endpoint", "tcp://localhost:5555",
            "--state-endpoint", "tcp://localhost:5557"]
    assert main(args) == 0
    capsys.readouterr()
    station = load_station(path)
    report = preflight(station, {"camera": {"ok": True}, "state": {"ok": True}})
    assert report["streams_ready"] is True
    assert report["motion_ready"] is False
    assert report["blockers"]
    with pytest.raises(FileExistsError):
        main(args)


@pytest.mark.parametrize("update", [{"hand": "unknown"}, {"camera_endpoint": "not-an-endpoint"},
                                     {"schema_version": 99}])
def test_malformed_station_fails_before_connecting(tmp_path, update):
    data = {"schema_version": 1, "robot_id": "test", "hand": "dex3",
            "camera_endpoint": "tcp://localhost:5555", "state_endpoint": "tcp://localhost:5557"}
    data.update(update)
    path = tmp_path / "station.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_station(path)


def test_preflight_does_not_promote_file_existence_to_calibration_or_motion(tmp_path):
    record = tmp_path / "unreviewed.json"
    record.write_text("{}")
    station = {"robot_id": "test", "hand": "dex3", "calibration_record": str(record),
               "operator_preflight_record": str(record)}
    report = preflight(station, {"camera": {"ok": True}, "state": {"ok": False}})
    assert report["streams_ready"] is False
    assert report["calibration"]["exists"] is True
    assert report["calibration"]["validated"] is False
    assert report["motion_ready"] is False


def test_g1_help_and_station_creation_need_no_camera_or_transport_packages(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"cv2", "zmq", "pyarrow", "pyrealsense2"}:
            raise AssertionError(f"Optional dependency imported for help/profile creation: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(SystemExit) as exit_info:
        main(["g1", "--help"])
    assert exit_info.value.code == 0
    assert "inspect-dataset" in capsys.readouterr().out
    path = tmp_path / "no-camera-packages.json"
    assert main(["g1", "init-station", "--output", str(path), "--robot-id", "test",
                 "--hand", "none", "--camera-endpoint", "tcp://localhost:5555",
                 "--state-endpoint", "tcp://localhost:5557"]) == 0
    assert load_station(path)["hand"] == "none"
