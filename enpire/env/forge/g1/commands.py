"""Lazy CLI entry points for G1 software preparation and observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .station import HANDS, load_station, preflight


def _emit(value: dict, output: Path | None) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


def _init_station(args: argparse.Namespace) -> int:
    value = {"schema_version": 1, "robot_id": args.robot_id, "hand": args.hand,
             "camera_endpoint": args.camera_endpoint, "state_endpoint": args.state_endpoint,
             "calibration_record": None, "operator_preflight_record": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing station and calibration references.
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    try:
        load_station(args.output)
    except ValueError:
        args.output.unlink()
        raise
    print(args.output)
    return 0


def _probe(args: argparse.Namespace) -> tuple[dict, dict]:
    from .transport import Subscriber

    station = load_station(args.station)
    observations = {}
    streams = ("camera",) if args.camera_only else ("camera", "state")
    for stream in streams:
        try:
            with Subscriber(station[f"{stream}_endpoint"], stream, hand=station["hand"]) as reader:
                values = reader.read_many(args.samples, args.timeout)
            summary = {"ok": True, "samples": len(values), "source_advances": True,
                       "first_timestamp": values[0]["timestamp"],
                       "last_timestamp": values[-1]["timestamp"],
                       "receive_span_seconds": values[-1]["received_monotonic"] - values[0]["received_monotonic"]}
            if stream == "camera":
                summary.update(image_shape=list(values[-1]["image"].shape),
                               image_dtype=str(values[-1]["image"].dtype), image_color_order="RGB",
                               depth_verified=False)
            else:
                summary["last_observation"] = {k: v for k, v in values[-1].items()
                                               if k != "received_monotonic"}
            observations[stream] = summary
        except (ImportError, ValueError, TimeoutError) as error:
            observations[stream] = {"ok": False, "error": str(error)}
    return station, observations


def _observe(args: argparse.Namespace) -> int:
    station, observations = _probe(args)
    _emit({"robot_id": station["robot_id"], "observations": observations,
           "motion_enabled": False, "cross_host_clock_sync_verified": False}, args.output)
    return 0 if all(v["ok"] for v in observations.values()) else 2


def _preflight(args: argparse.Namespace) -> int:
    station, observations = _probe(args)
    _emit(preflight(station, observations) | {"observations": observations}, args.output)
    return 2  # Physical preflight cannot pass until the missing physical gates are implemented.


def _inspect_dataset(args: argparse.Namespace) -> int:
    from .dataset import inspect_dataset

    try:
        report = inspect_dataset(args.dataset)
    except (ImportError, OSError, ValueError) as error:
        _emit({"ok": False, "error": str(error), "motion_enabled": False}, args.output)
        return 2
    _emit(report, args.output)
    return 0


def add_g1_parser(commands) -> None:
    parser = commands.add_parser("g1", help="Inspect G1 SONIC streams and datasets without motion")
    actions = parser.add_subparsers(dest="g1_command", required=True)
    init = actions.add_parser("init-station", help="Create a new external station JSON")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--robot-id", required=True)
    init.add_argument("--hand", choices=HANDS, required=True)
    init.add_argument("--camera-endpoint", required=True)
    init.add_argument("--state-endpoint", required=True)
    init.set_defaults(handler=_init_station)
    for name, handler in (("observe", _observe), ("preflight", _preflight)):
        action = actions.add_parser(name, help="Read streams" if name == "observe" else "Report outstanding physical gates")
        action.add_argument("--station", type=Path, required=True)
        action.add_argument("--samples", type=int, default=5)
        action.add_argument("--timeout", type=float, default=15.0, help="Total deadline per stream in seconds")
        action.add_argument("--camera-only", action="store_true")
        action.add_argument("--output", type=Path)
        action.set_defaults(handler=handler)
    dataset = actions.add_parser("inspect-dataset", help="Validate offline LeRobot handover data")
    dataset.add_argument("--dataset", type=Path, required=True)
    dataset.add_argument("--output", type=Path)
    dataset.set_defaults(handler=_inspect_dataset)
