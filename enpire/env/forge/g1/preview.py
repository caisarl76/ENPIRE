"""Offline pose inspection; no controller, transport, or dynamics stepping."""

from __future__ import annotations


def map_qpos(names, state, model_names, addresses, base_qpos):
    """Map every recorded joint by name, preserving the nominal free-base pose."""
    import numpy as np

    names, model_names = list(names), list(model_names)
    if (len(set(names)) != len(names) or len(set(model_names)) != len(model_names)
            or set(names) != set(model_names)):
        raise ValueError("Recorded and model joint names must match uniquely and completely")
    if len(addresses) != len(model_names) or len(set(addresses)) != len(addresses):
        raise ValueError("Model qpos addresses must be unique and complete")
    reference = np.asarray(base_qpos, dtype=float)
    state = np.asarray(state, dtype=float)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("Nominal base configuration must be a finite vector")
    if state.ndim != 2 or state.shape[1] != len(names) or not len(state) or not np.isfinite(state).all():
        raise ValueError("Recorded state must be a finite frame-by-joint array")
    if any(not isinstance(a, (int, np.integer)) or a < 0 or a >= len(reference) for a in addresses):
        raise ValueError("Model qpos addresses must be in bounds")
    address_by_name = dict(zip(model_names, addresses))
    result = np.tile(reference, (len(state), 1))
    result[:, [address_by_name[name] for name in names]] = state
    return result


def limit_report(names, state, lower, upper, tolerance=1e-5):
    """Report raw out-of-model values without changing the recorded poses."""
    import numpy as np

    state = np.asarray(state, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (state.ndim != 2 or not len(state) or state.shape[1] != len(names)
            or lower.shape != (len(names),) or upper.shape != lower.shape
            or not np.isfinite(state).all() or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Limit inspection requires finite state and ordered joint bounds")
    excess = np.maximum(np.maximum(lower - state, state - upper), 0)
    outside = excess > tolerance
    return {"tolerance_rad": tolerance,
            "frames_outside_model_limits": int(np.any(outside, axis=1).sum()),
            "frames_exceeding_0_01_rad": int(np.any(excess > 0.01, axis=1).sum()),
            "joints": [{"name": name, "min_rad": float(state[:, i].min()),
                        "max_rad": float(state[:, i].max()),
                        "lower_rad": float(lower[i]), "upper_rad": float(upper[i]),
                        "frames_outside": int(outside[:, i].sum()),
                        "max_excess_rad": float(excess[:, i].max())}
                       for i, name in enumerate(names)]}


def load_bundle(path):
    """Validate a numeric pose bundle before optional model/graphics imports."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    schema = bundle.get("schema_version", np.asarray(0))
    if schema.shape != () or schema != 1:
        raise ValueError("Unsupported preview bundle schema")
    fps = bundle.get("fps", np.asarray(0))
    if fps.shape != () or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Preview fps must be positive and finite")
    names = bundle.get("joint_names", np.asarray([]))
    if names.shape != (43,) or names.dtype.kind != "U" or len(set(names)) != 43:
        raise ValueError("Preview requires 43 unique joint names")
    state = bundle.get("state", np.asarray([]))
    if state.ndim != 2 or state.shape[1] != 43 or not len(state):
        raise ValueError("Preview state must contain frames of 43 joints")
    length = len(state)
    for key, shape in (("state", (length, 43)), ("eef_state", (length, 14)),
                       ("root_orientation", (length, 4)), ("timestamp", (length,)),
                       ("episode_index", (length,)), ("frame_index", (length,)),
                       ("stream_mode", (length,))):
        array = bundle.get(key, np.asarray([]))
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Preview {key} requires finite shape {shape}")
    for key in ("episode_index", "frame_index", "stream_mode"):
        array = bundle[key]
        if np.any(array != np.floor(array)) or np.any(array < 0):
            raise ValueError(f"Preview {key} must contain nonnegative integers")
        bundle[key] = array.astype(np.int64)
    if not np.isin(bundle["stream_mode"], np.arange(6)).all():
        raise ValueError("Invalid preview stream mode")
    quat = bundle["root_orientation"]
    if np.any(np.abs(np.linalg.norm(quat, axis=1) - 1) > 0.01):
        raise ValueError("Preview root orientation must be unit wxyz quaternions")
    for offset in (3, 10):
        if np.any(np.abs(np.linalg.norm(bundle["eef_state"][:, offset:offset+4], axis=1) - 1) > 0.01):
            raise ValueError("Preview EEF orientation must be unit wxyz quaternions")
    episodes = np.unique(bundle["episode_index"])
    if not np.array_equal(episodes, np.arange(len(episodes))) or np.any(np.diff(bundle["episode_index"]) < 0):
        raise ValueError("Preview episodes must be contiguous and ordered")
    for episode in episodes:
        selected = bundle["episode_index"] == episode
        if not np.array_equal(bundle["frame_index"][selected], np.arange(selected.sum())):
            raise ValueError("Preview frame indices must be contiguous within each episode")
        if np.any(np.diff(bundle["timestamp"][selected]) <= 0):
            raise ValueError("Preview timestamps must advance within each episode")
    videos = bundle.get("video_paths", np.asarray([]))
    if videos.shape != (len(episodes),) or videos.dtype.kind != "U":
        raise ValueError("Preview requires one video-path string per episode")
    return bundle


def _model_joints(model):
    import mujoco
    import numpy as np

    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(free) != 1 or len(hinges) + 1 != model.njnt or not model.jnt_limited[hinges].all():
        raise ValueError("Preview model requires one free base and limited hinge joints")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in hinges]
    return names, model.jnt_qposadr[hinges], hinges, int(model.jnt_qposadr[free[0]])


def inspect_poses(bundle, model):
    """Inspect all raw poses with mj_forward, without taking any dynamics steps."""
    import mujoco
    import numpy as np

    names = list(bundle["joint_names"])
    model_names, addresses, hinges, free_address = _model_joints(model)
    reference = model.qpos0.copy()
    reference[free_address:free_address+7] = [0, 0, 0, 1, 0, 0, 0]
    qpos = map_qpos(names, bundle["state"], model_names, addresses, reference)
    joint_ids = [int(hinges[model_names.index(name)]) for name in names]
    bounds = model.jnt_range[joint_ids]
    limits = limit_report(names, bundle["state"], bounds[:, 0], bounds[:, 1])
    data = mujoco.MjData(model)
    bodies = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link")
              for side in ("left", "right")}
    if any(body < 0 for body in bodies.values()):
        raise ValueError("Preview model requires left and right wrist-yaw bodies")
    errors = {side: {"position": [], "rotation": []} for side in bodies}
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for side, offset in (("left", 0), ("right", 7)):
            observed = bundle["eef_state"][frame, offset:offset+7]
            body = bodies[side]
            errors[side]["position"].append(float(np.linalg.norm(data.xpos[body] - observed[:3])))
            quat = observed[3:] / np.linalg.norm(observed[3:])
            cosine = np.clip(abs(np.dot(data.xquat[body], quat)), 0, 1)
            errors[side]["rotation"].append(float(np.degrees(2 * np.arccos(cosine))))
    fk = {side: {"max_position_error_m": max(value["position"]),
                 "p95_position_error_m": float(np.percentile(value["position"], 95)),
                 "frames_over_5mm": int(np.count_nonzero(np.asarray(value["position"]) > 0.005)),
                 "max_rotation_error_deg": max(value["rotation"])}
          for side, value in errors.items()}
    episodes = []
    for episode in np.unique(bundle["episode_index"]):
        selected = bundle["episode_index"] == episode
        partial = limit_report(names, bundle["state"][selected], bounds[:, 0], bounds[:, 1])
        episodes.append({"episode_index": int(episode), "frames": int(selected.sum()),
                         "duration_seconds": float(selected.sum() / bundle["fps"]),
                         "frames_outside_model_limits": partial["frames_outside_model_limits"],
                         "frames_exceeding_0_01_rad": partial["frames_exceeding_0_01_rad"]})
    return {"schema_version": 1, "frames": len(qpos), "episodes": episodes,
            "joint_qpos_addresses": dict(zip(names, map(int, [addresses[model_names.index(n)] for n in names]))),
            "limits": limits, "eef_fk": fk, "eef_comparison_frame": "pelvis, identity base orientation",
            "motion_enabled": False, "root_translation_measured": False,
            "image_state_alignment_verified": False,
            "raw_joints_clipped": False, "balance_verified": False,
            "collision_verified": False, "physical_success_verified": False,
            "limitations": [
                "Kinematic inspection does not test controller behavior, contacts, or grasp retention",
                "Root translation is unavailable; rendering uses the model's nominal base position",
                "Exporter EEF values may reflect supplemental joint clamping; raw pose FK may differ",
                "Model revision, palm axes, camera calibration, and station geometry remain unverified",
            ]}


def render_preview(bundle, model, episode, output, stride=2):
    """Render one recorded episode beside its RGB video at recording speed."""
    from pathlib import Path

    import cv2
    import mujoco
    import numpy as np

    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Preview stride must be a positive integer")
    selected = np.flatnonzero(bundle["episode_index"] == episode)
    if not len(selected):
        raise ValueError("Requested preview episode does not exist")
    source = Path(str(bundle["video_paths"][episode]))
    if not source.is_file():
        raise FileNotFoundError("Recorded RGB video is required for a paired preview")
    output = Path(output)
    if output.exists() or output.with_suffix(".png").exists():
        raise FileExistsError(output)
    model_names, addresses, _, free_address = _model_joints(model)
    qpos = map_qpos(bundle["joint_names"], bundle["state"][selected], model_names, addresses, model.qpos0)
    qpos[:, free_address+3:free_address+7] = bundle["root_orientation"][selected]
    norms = np.linalg.norm(qpos[:, free_address+3:free_address+7], axis=1, keepdims=True)
    qpos[:, free_address+3:free_address+7] /= norms
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = model.qpos0[free_address:free_address+3] + [0.0, 0.0, 0.15]
    camera.distance, camera.azimuth, camera.elevation = 2.3, 135, -12
    hand_camera = mujoco.MjvCamera()
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.distance, hand_camera.azimuth, hand_camera.elevation = 0.6, 135, -25
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if wrist < 0:
        raise ValueError("Preview model requires a right wrist-yaw body")
    model.vis.headlight.ambient[:] = 0.6
    model.vis.headlight.diffuse[:] = 0.8
    reader = cv2.VideoCapture(str(source))
    if not reader.isOpened():
        reader.release()
        raise ValueError("Recorded RGB video could not be opened")
    renderer = None
    writer = None
    sampled = list(range(0, len(selected), stride))
    snapshots = []
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                 float(bundle["fps"] / stride), (1280, 600))
        if not writer.isOpened():
            raise ValueError("Preview AVI encoder could not be opened")
        snapshot_indices = set(np.linspace(0, len(sampled)-1, 4, dtype=int))
        for count, local_frame in enumerate(sampled):
            frame = int(selected[local_frame])
            wanted = int(bundle["frame_index"][frame])
            while reader.get(cv2.CAP_PROP_POS_FRAMES) < wanted:
                if not reader.grab():
                    raise ValueError(f"RGB video ended before source frame {wanted}")
            ok, rgb_bgr = reader.read()
            if not ok:
                raise ValueError(f"RGB frame {wanted} could not be decoded")
            data.qpos[:] = qpos[local_frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            overview = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            hand_camera.lookat[:] = data.xpos[wrist] + 0.05 * data.xmat[wrist].reshape(3, 3)[:, 0]
            renderer.update_scene(data, camera=hand_camera)
            detail = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            overview[240:480, 320:640] = cv2.resize(detail, (320, 240))
            canvas = np.full((600, 1280, 3), [29, 24, 19], dtype=np.uint8)
            canvas[48:528, :640] = cv2.resize(rgb_bgr, (640, 480))
            canvas[48:528, 640:] = overview
            cv2.putText(canvas, "Recorded head-camera RGB", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)
            cv2.putText(canvas, "G1 recorded pose + right-hand detail", (656, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
            stamp = float(bundle["timestamp"][frame])
            cv2.putText(canvas, f"Episode {episode} | source frame {wanted} | t={stamp:.2f}s | pose inspection only",
                        (16, 554), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
            cv2.putText(canvas, "Root position fixed at model reference; object/fixture absent from model; no controller or motion commands",
                        (16, 581), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (150, 205, 250), 1)
            writer.write(canvas)
            if count in snapshot_indices:
                snapshots.append(cv2.resize(canvas, (640, 300)))
        if len(snapshots) == 4:
            sheet = np.vstack([np.hstack(snapshots[:2]), np.hstack(snapshots[2:])])
            if not cv2.imwrite(str(output.with_suffix(".png")), sheet):
                raise OSError("Could not write preview contact sheet")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
    return {"episode_index": int(episode), "video": str(output), "frames": len(sampled),
            "fps": float(bundle["fps"] / stride), "source_video": str(source),
            "source_frame_indices": [int(bundle["frame_index"][selected[i]]) for i in sampled],
            "nominal_root_position_m": model.qpos0[free_address:free_address+3].tolist(),
            "root_orientation": "recorded wxyz, normalized for floating-point roundoff",
            "motion_enabled": False}


def main(argv=None):
    """Standalone entry for existing MuJoCo Python environments, including 3.10."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validated export-preview NPZ")
    parser.add_argument("--model", type=Path, required=True, help="Local G1 MuJoCo XML")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--render-episode", type=int)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle(args.input)
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        report = inspect_poses(bundle, model)
        report.update(model=str(args.model.resolve()), mujoco_version=mujoco.__version__,
                      model_xml_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      bundle=str(args.input.resolve()))
        args.output.mkdir(parents=True, exist_ok=False)
        # Save the audit before rendering so inspection evidence survives graphics failures.
        target = args.output / "validation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.render_episode is not None:
            report["render"] = render_preview(bundle, model, args.render_episode,
                                               args.output / f"episode_{args.render_episode:06d}.avi",
                                               stride=args.stride)
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"report": str(target), "frames": report["frames"],
                          "frames_outside_model_limits": report["limits"]["frames_outside_model_limits"],
                          "motion_enabled": False}, indent=2))
        return 0
    except (ImportError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error), "motion_enabled": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
