"""External G1 station records and conservative physical-preflight reporting."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

HANDS = ("none", "dex3", "inspire_ftp")
CAMERA_PROTOCOLS = ("sonic_msgpack", "json_rgb_jpeg")
_FIELDS = {
    "schema_version", "robot_id", "hand", "camera_endpoint", "state_endpoint",
    "calibration_record", "operator_preflight_record", "camera_protocol", "camera_key",
}


def load_station(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.keys() - _FIELDS:
        raise ValueError("Station must be an object containing only documented station fields")
    if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        raise ValueError("Expected station schema_version 1")
    if not isinstance(data.get("robot_id"), str) or not data["robot_id"].strip():
        raise ValueError("Station requires a nonempty robot_id")
    if data.get("hand") not in HANDS:
        raise ValueError(f"Station hand must be one of {HANDS}")
    data.setdefault("camera_protocol", "sonic_msgpack")
    data.setdefault("camera_key", "ego_view")
    if data["camera_protocol"] not in CAMERA_PROTOCOLS:
        raise ValueError(f"camera_protocol must be one of {CAMERA_PROTOCOLS}")
    if not isinstance(data["camera_key"], str) or not data["camera_key"].strip():
        raise ValueError("camera_key must be a nonempty string")
    for key in ("camera_endpoint", "state_endpoint"):
        value = data.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Station requires {key}")
        address = urlsplit(value)
        if (address.scheme != "tcp" or not address.hostname or not address.port
                or address.path or address.query or address.fragment
                or address.username or address.password):
            raise ValueError(f"{key} must be a TCP host:port endpoint without credentials")
    for key in ("calibration_record", "operator_preflight_record"):
        value = data.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a path or null")
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = Path(path).resolve().parent / resolved
            data[key] = str(resolved.resolve())
    return data


def preflight(station: dict, observations: dict) -> dict:
    def record(key: str) -> dict:
        path = station.get(key)
        return {"path": path, "exists": bool(path and Path(path).is_file()), "validated": False}

    streams_ready = all(observations.get(stream, {}).get("ok") is True
                        for stream in ("camera", "state"))
    blockers = []
    if not streams_ready:
        blockers.append("Advancing camera and SONIC state streams have not both been verified")
    if observations.get("state", {}).get("source_timestamp_available") is False:
        blockers.append("SONIC state has no source timestamp; camera/state timing needs independent validation")
    if station["hand"] != "dex3":
        blockers.append("This bottle task requires a fitted, characterized hand; only Dex3 mapping is prepared")
    blockers.extend([
        "Camera intrinsics, camera mounting transform, torso/waist kinematics, and fixture pose require validation",
        "RGB-only camera requires a validated depth or surface-plane solution for metric grasp targets",
        "Operator must complete station, emergency-stop, support, joint-limit, and hand preflight",
        "SONIC command ownership and manipulation bridge require simulation validation",
        "Independent presentation/recipient-acquisition verifier and safe reset require implementation",
        "Explicit motion authorization is required for the specific robot and trial",
    ])
    return {
        "robot_id": station["robot_id"], "hand": station["hand"],
        "streams_ready": streams_ready, "motion_ready": False,
        "calibration": record("calibration_record"),
        "operator_preflight": record("operator_preflight_record"),
        "blockers": blockers,
        "note": "Record existence and observation liveness are not calibration or motion authorization",
    }
