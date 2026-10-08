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
    {"ros_timestamp": -1.0},
    {"ros_timestamp": None},
])
def test_invalid_telemetry_cannot_be_used_as_an_observation(updates):
    with pytest.raises(ValueError):
        decode_state(telemetry(**updates), hand="dex3")


def test_state_rejects_wrong_topic():
    with pytest.raises(ValueError):
        decode_state(b"robot_config" + msgpack.packb({}), hand="dex3")


def test_zero_ros_timestamp_is_explicitly_unavailable_not_a_fabricated_time():
    observation = decode_state(telemetry(ros_timestamp=0.0), hand="dex3")
    assert observation["timestamp"] is None
    assert observation["raw_ros_timestamp"] == 0.0
    assert observation["source_timestamp_available"] is False
    assert observation["timestamp_clock"] is None


@pytest.fixture
def telemetry_source():
    """Exercise the real subscription transport with caller-provided source data."""
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    threads = []

    def start(message):
        def send():
            index = 1
            while not stop.wait(0.01):
                publisher.send(message(index))
                index += 1

        thread = threading.Thread(target=send)
        thread.start()
        threads.append(thread)
        return f"tcp://127.0.0.1:{port}"

    yield start
    stop.set()
    for thread in threads:
        thread.join()
    publisher.close(linger=0)
    context.term()


def test_live_state_with_unavailable_ros_time_uses_advancing_real_indices(telemetry_source):
    endpoint = telemetry_source(lambda index: telemetry(index=index, ros_timestamp=0.0))
    with Subscriber(endpoint, "state", hand="dex3") as reader:
        samples = reader.read_many(3, 2.0)
    assert len(samples) == 3
    assert all(a["index"] < b["index"] for a, b in zip(samples, samples[1:]))
    assert all(sample["timestamp"] is None for sample in samples)


def test_repeated_zero_timestamp_state_index_still_fails_liveness(telemetry_source):
    endpoint = telemetry_source(lambda index: telemetry(index=1, ros_timestamp=0.0))
    with Subscriber(endpoint, "state", hand="dex3") as reader:
        with pytest.raises(TimeoutError):
            reader.read_many(2, 0.3)


def test_positive_stalled_ros_time_is_not_treated_as_unavailable(telemetry_source):
    endpoint = telemetry_source(lambda index: telemetry(index=index, ros_timestamp=100.0))
    with Subscriber(endpoint, "state", hand="dex3") as reader:
        with pytest.raises(TimeoutError):
            reader.read_many(2, 0.3)


def test_state_timestamp_availability_change_requires_a_new_probe(telemetry_source):
    # Alternate so the first received message does not depend on PUB/SUB startup.
    endpoint = telemetry_source(lambda index: telemetry(index=index, ros_timestamp=0.0 if index % 2 else 100.0 + index))
    with Subscriber(endpoint, "state", hand="dex3") as reader:
        with pytest.raises(ValueError, match="timestamp availability changed"):
            reader.read_many(3, 2.0)


@pytest.mark.parametrize("changed_index", [9, 10])
def test_timestamp_availability_change_cannot_hide_behind_old_indices(changed_index):
    messages = iter([telemetry(index=10, ros_timestamp=0.0),
                     telemetry(index=changed_index, ros_timestamp=100.0),
                     telemetry(index=11, ros_timestamp=0.0)])

    class RawSocket:
        def poll(self, timeout_ms):
            return True

        def recv(self):
            return next(messages)

    reader = Subscriber.__new__(Subscriber)
    reader.stream, reader.hand, reader._socket = "state", "dex3", RawSocket()
    with pytest.raises(ValueError, match="timestamp availability changed"):
        reader.read_many(2, 1.0)


def test_cli_reports_sequence_liveness_and_missing_source_time(tmp_path, monkeypatch, capsys):
    path = tmp_path / "station.json"
    path.write_text(json.dumps({"schema_version": 1, "robot_id": "test", "hand": "dex3",
                               "camera_endpoint": "tcp://localhost:5555", "state_endpoint": "tcp://localhost:5557"}))

    class RecordedSubscriber:
        def __init__(self, endpoint, stream, **kwargs):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read_many(self, samples, timeout):
            if self.stream == "state":
                values = [decode_state(telemetry(index=index, ros_timestamp=0.0), hand="dex3") for index in range(samples)]
            else:
                values = [{"timestamp": 100.0 + index, "image": np.zeros((4, 6, 3), dtype=np.uint8)} for index in range(samples)]
            return [value | {"received_monotonic": float(index)} for index, value in enumerate(values)]

    monkeypatch.setattr("enpire.env.forge.g1.transport.Subscriber", RecordedSubscriber)
    assert main(["g1", "observe", "--station", str(path)]) == 0
    state = json.loads(capsys.readouterr().out)["observations"]["state"]
    assert state["ok"] is True
    assert state["source_timestamp_available"] is False
    assert state["first_timestamp"] is None and state["last_timestamp"] is None
    assert state["first_index"] == 0 and state["last_index"] == 4
    assert state["liveness_basis"] == "source_index"
    assert state["last_observation"]["raw_ros_timestamp"] == 0.0
    assert main(["g1", "preflight", "--station", str(path)]) == 2
    preflight_report = json.loads(capsys.readouterr().out)
    assert preflight_report["streams_ready"] is True
    assert preflight_report["motion_ready"] is False
    assert any("source timestamp" in blocker for blocker in preflight_report["blockers"])


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


def rgb_json_camera(index=1, *, color=(255, 0, 0)):
    """Match the active publisher: JPEG-encode RGB, decode without swapping."""
    cv2 = pytest.importorskip("cv2")
    rgb = np.full((8, 8, 3), color, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", rgb)
    assert ok
    return json.dumps({"images": {"egocentric": base64.b64encode(jpeg).decode()},
                       "timestamps": {"egocentric": 123.0 + index / 30}}).encode()


@pytest.mark.parametrize("color", [(255, 0, 0), (0, 0, 255)])
def test_json_rgb_camera_preserves_red_and_blue_channels(color):
    observation = decode_camera(rgb_json_camera(color=color),
                                protocol="json_rgb_jpeg", key="egocentric")
    assert observation["image"].shape == (8, 8, 3)
    np.testing.assert_allclose(observation["image"][0, 0], color, atol=5)
    assert observation["timestamp"] == pytest.approx(123.0 + 1 / 30)


def test_camera_protocol_and_key_are_explicit_without_fallback():
    raw = rgb_json_camera()
    with pytest.raises(ValueError):
        decode_camera(raw)
    with pytest.raises(ValueError):
        decode_camera(raw, protocol="json_rgb_jpeg", key="ego_view")
    with pytest.raises(ValueError):
        decode_camera(raw, protocol="guess", key="egocentric")


@pytest.mark.parametrize("timestamp", [None, True, math.nan])
def test_json_camera_rejects_invalid_source_time(timestamp):
    data = json.loads(rgb_json_camera())
    data["timestamps"]["egocentric"] = timestamp
    with pytest.raises(ValueError):
        decode_camera(json.dumps(data).encode(), protocol="json_rgb_jpeg", key="egocentric")


def test_cli_reads_configured_json_camera_without_a_controller(tmp_path, telemetry_source, capsys):
    endpoint = telemetry_source(rgb_json_camera)
    path = tmp_path / "station.json"
    path.write_text(json.dumps({"schema_version": 1, "robot_id": "test", "hand": "dex3",
                               "camera_endpoint": endpoint, "state_endpoint": "tcp://localhost:5557",
                               "camera_protocol": "json_rgb_jpeg", "camera_key": "egocentric"}))
    assert main(["g1", "observe", "--station", str(path), "--camera-only",
                 "--samples", "3", "--timeout", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["motion_enabled"] is False
    camera = report["observations"]["camera"]
    assert camera["source_advances"] is True
    assert camera["samples"] == 3
    assert camera["camera_protocol"] == "json_rgb_jpeg"
    assert camera["camera_key"] == "egocentric"
    assert camera["image_color_order"] == "RGB"


def test_legacy_station_keeps_composed_camera_defaults(tmp_path):
    path = tmp_path / "station.json"
    path.write_text(json.dumps({"schema_version": 1, "robot_id": "test", "hand": "dex3",
                               "camera_endpoint": "tcp://localhost:5555",
                               "state_endpoint": "tcp://localhost:5557"}))
    station = load_station(path)
    assert station["camera_protocol"] == "sonic_msgpack"
    assert station["camera_key"] == "ego_view"


@pytest.mark.parametrize("update", [{"camera_protocol": "guess"}, {"camera_protocol": None},
                                     {"camera_key": ""}, {"camera_key": True}])
def test_invalid_camera_configuration_fails_before_connecting(tmp_path, update):
    path = tmp_path / "station.json"
    data = {"schema_version": 1, "robot_id": "test", "hand": "dex3",
            "camera_endpoint": "tcp://localhost:5555", "state_endpoint": "tcp://localhost:5557"}
    path.write_text(json.dumps(data | update))
    with pytest.raises(ValueError):
        load_station(path)


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
