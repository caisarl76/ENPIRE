# G1 observations and bottle-handover preparation

GEAR-SONIC continues to own balance, locomotion, teleoperation, and robot safety.
The `enpire g1` commands subscribe to its camera and telemetry streams. They do
not launch a controller or publish commands. The existing YAM gravity-comp and
calibration commands in [REAL_WORLD_WORKFLOWS.md](REAL_WORLD_WORKFLOWS.md) target
YAM hardware; G1 requires its own characterized station and calibration workflow.

## Install and create external station records

Use an existing configured Python 3.11 environment when available. A standalone
client installation needs the `g1` extra; offline Parquet inspection also needs
`g1-dataset`. When using `uv sync`, pass all existing required extras together,
because sync removes packages outside the selected dependency set.

```bash
git submodule update --init --recursive
uv sync --extra dev --extra g1 --extra g1-dataset
enpire g1 --help
```

On Jetson, preserve the existing camera SDK environment and consume its stream.
The root pinned RealSense Python 3.11 wheel and some generic planning dependencies
do not provide compatible ARM wheels. The G1 observation client needs neither
the RealSense SDK nor the YAM planning stack.

Keep robot-specific endpoints, identities, paths, and records outside Git:

```bash
enpire g1 init-station --output "$STATION_JSON" --robot-id "$ROBOT_ID" \
  --hand dex3 --camera-endpoint "$CAMERA_ENDPOINT" --state-endpoint "$STATE_ENDPOINT"
```

Use `--hand none` for an unfitted hand and `--hand inspire_ftp` only once that
hardware is attached. The command refuses to overwrite an existing record.
Set `calibration_record` and `operator_preflight_record` to external files when
available. Relative record paths resolve against the station JSON directory.
Presence of a file is reported separately from validation.

## Read actual camera frames and SONIC state

In the existing GR00T-WholeBodyControl camera environment, the camera server is:

```bash
source .venv_camera/bin/activate
python gear_sonic/camera/composed_camera.py --ego-view-camera realsense --port 5555
```

This command currently supplies RGB JPEG frames, not validated metric depth.
Do not restart an existing camera server just to run observation tools.

The station's optional `camera_protocol` and `camera_key` fields select the
camera wire format explicitly. Existing records default to `sonic_msgpack` and
`ego_view`, preserving the composed-camera protocol. For a running JSON/base64
publisher that JPEG-encodes RGB arrays and expects no channel swap on decode,
set `camera_protocol` to `json_rgb_jpeg` and `camera_key` to its actual stream
name in the external station record. This reader preserves the JPEG-decoded RGB
channels. The MessagePack reader instead converts normal decoded BGR JPEGs to
RGB. Choosing the correct protocol includes choosing the color convention;
JSON and MessagePack are not interchangeable merely by renaming the stream.
The reader does not auto-detect a protocol or select a different camera key.
Both modes require advancing finite source timestamps and valid JPEG images.
Observation reports include the selected protocol/key. Stream delivery remains
separate from camera identity, calibration, and cross-host timing validation.

```bash
enpire g1 observe --station "$STATION_JSON" --camera-only --output "$CAMERA_REPORT"
enpire g1 observe --station "$STATION_JSON" --output "$OBSERVATION_REPORT"
enpire g1 preflight --station "$STATION_JSON" --output "$PREFLIGHT_REPORT"
```

The default is five advancing samples with a 15-second total deadline per stream.
Missing, malformed, or stalled streams fail. `observe --camera-only` can pass
while SONIC is offline. Full observation requires its existing `g1_debug` state
publisher; the tool never starts it. Camera reports describe decoded RGB images.
The installed SONIC controller may send `ros_timestamp: 0` when no ROS2 output
clock is available. In that case the observation exposes `timestamp: null`,
preserves `raw_ros_timestamp: 0`, and reports `source_timestamp_available: false`.
Advancing measured-state sequence indices establish subscription liveness;
they do not provide source timing. Repeated indices still fail. With positive
ROS timestamps, both index and timestamp must advance; a change in timestamp
availability requires a new probe. No source time is synthesized from sequence
indices or local receipt clocks. Camera/state alignment remains unverified,
and preflight explicitly lists missing state source timing as a blocker.
Dex3 state interleaves the left hand before the right arm to match the dataset's
43-joint layout. Raw `left_hand_q` and `right_hand_q` preserve SONIC's Dex3 motor
order (thumb 0/1/2, index 0/1, middle 0/1). The combined `state` reorders each hand
to the dataset layout (index 0/1, middle 0/1, thumb 0/1/2).
Other hand types report the measured 29-joint body only.
No unmeasured hand or end-effector pose is filled with artificial zeros.

The pinned ZMQ publisher restores `default_angles` and hardware body ordering
before sending `body_q`. Use the published angles directly for joint kinematics.
The controller's internal logger stores relative angles in IsaacLab ordering;
that internal representation differs from the outgoing message.

Receipt liveness does not prove cross-host clock synchronization or calibrated
image/state alignment. `preflight` exits with code 2 and `motion_ready: false`
until the outstanding physical and control gates have been completed.

## Inspect the task demonstrations offline

```bash
enpire g1 inspect-dataset --dataset "$DATASET_ROOT" --output "$DATASET_REPORT"
```

The reader checks LeRobot v2.1 episode/frame counts, required feature sizes,
finite values, stream modes, and exact Dex3 joint ordering. It reports arm/hand
movement ranges and mode counts. It does not mutate the dataset, replay actions,
or assign grasp/handover success labels. Review RGB videos alongside this report
to establish which hand and task phases the demonstrations actually show.

### Export and preview recorded poses

The workstation can export a pickle-free NPZ containing all validated episodes:

```bash
enpire g1 export-preview --dataset "$DATASET_ROOT" --output "$PREVIEW_NPZ"
```

Export validates joint order, episode/frame identity, finite values, advancing
timestamps, and root/EEF unit quaternions. The output must be a new file. It
includes no fabricated root translation or command packets.

Use an existing simulation Python with MuJoCo, NumPy, and OpenCV to run the
standalone preview script. This script supports Python 3.10+ independently of
the ENPIRE Python 3.11 installation. For CPU rendering on a Linux workstation
with Mesa EGL, configure the existing interpreter and Mesa vendor JSON paths:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless \
  __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL_VENDOR_JSON" \
  "$MUJOCO_PYTHON" "$ENPIRE_ROOT/enpire/env/forge/g1/preview.py" \
  --input "$PREVIEW_NPZ" --model "$G1_MODEL_XML" --output "$PREVIEW_DIR" \
  --render-episode 0 --stride 2
```

The artifact directory must be new. All poses are mapped to model joints by
name and inspected with `mj_forward`; no dynamics steps or transport sockets
are used. `validation.json` reports model-limit excess and raw joint FK versus
recorded EEF values. The paired AVI and PNG show saved RGB beside the pose model.
Stride 2 produces a 25 fps preview of a 50 fps recording at its recorded speed.
Rendered root position stays at the model's nominal reference; root quaternion
comes from the recording. Object and fixture geometry are absent from the model.

The exporter uses supplemental joint clamping when computing EEF features.
Those EEF values can differ from forward kinematics of raw joint observations,
particularly near the shoulder-roll boundary. The preview preserves raw angles
and reports that discrepancy. Neither a visually plausible pose nor a matching
EEF value establishes calibrated image/state alignment, balance, collision
clearance, grasp retention, handover success, or physical motion readiness.

### Qualify the task adapter without a robot

The selected bottle-task architecture is a staged ENPIRE supervisor around a
SONIC motion-token actor. The actor consumes fresh images/state plus a phase/goal
and returns one 64-value whole-body token and two Dex3 hand targets. SONIC keeps
low-level body and hand control. Tokens can affect leg behavior, so the task
must exclusively own the selected SONIC input session. A compatible learned
policy or validated pose-to-token encoder is still needed for autonomous picking.

The socket-free reference adapter reuses the existing local SONIC byte builder:

```bash
enpire g1 shadow-adapter --dataset "$DATASET_ROOT" \
  --encoder "$SONIC_ROOT/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py" \
  --output "$NEW_QUALIFICATION_JSON"
```

Token-only packets require protocol **v4**, with a 1,280-byte header, token_state
float32 [1,64], frame_index int64 [1], and native thumb/index/middle Dex3 targets
float32 [1,7]. Version 3 instead requires body/SMPL/joint motion fields. A byte
round-trip of token fields under v3 does not establish receiver compatibility.

The reference actor uses saved POSE samples. The runner uses synthetic local
receipt clocks, models exclusive writer leases in memory, blocks standard-library
socket creation, and independently checks the v4 packet fields. It stops each
reference session at a mode or episode boundary and never saves command packets.
Its result does not verify image/state synchronization or a live receiver.

OfflineAdapter requires two advancing samples from each source before start.
Repeated source data cannot refresh liveness. Its watchdog and output checks
latch faults on stale state/camera/actor inputs, source resets, lost ownership,
invalid targets, and encoder errors. Data age and ownership are checked again
after encoding. Overlapping outputs latch a fault; stop can cancel an output
still being encoded. Real runners must call tick even when the actor is silent.
Explicit stop and faults end the session; resumption requires a new instance
and explicit start. Software timing limits are qualification parameters.

Stopping local output does **not** clear receiver-held tokens or establish a
safe robot fallback. Receiver disconnect behavior, real publisher arbitration,
policy/checkpoint compatibility, hand limits, transition ramps, balance, and
reset remain prerequisites before a motion-capable transport is implemented.
This command has no transport endpoint or motion-enable option.

The intended handover phases are object localization, approach, grasp, lift,
palm-up presentation, recipient acquisition, and retreat. A grasp must remain
supported while preparing the presentation. Finger spreading/release needs
independent confirmation that the bottle remains supported or the recipient has
acquired it. A palm-up terminal pose alone does not prove handover success.

Recorded `action.wbc` contains leg commands as well as arms and hands; it is not
a manipulation-only command to feed into SONIC. Recorded VR rotations use three
6D rotations (18 values), while the command protocol's quaternion fields use a
different representation. A future adapter must reuse existing SONIC encoders,
limits, timeout handling, and command-ownership mechanisms and characterize the
chosen control mode in simulation before hardware trials.

## Required physical gates

Before any future motion command, complete the station checks and calibration
required by [REAL_WORLD_WORKFLOWS.md](REAL_WORLD_WORKFLOWS.md): verify identity,
hand mounting and joint conventions, support/balance arrangement, limits,
collision-free workspace, operator presence, and the emergency-stop procedure.
Validate camera intrinsics and distortion, camera mounting geometry, torso/waist
kinematics, camera/state timing, and fixture pose with measured residuals
and a physical sanity check. Box dimensions alone do not establish its robot
frame pose. RGB needs a validated surface plane or depth source for 3D targets.

Then validate a manipulation adapter, reset, independent success verifier, and
SONIC control ownership in simulation. Finally obtain explicit motion
authorization naming the robot and trial. ENPIRE's observation commands cannot
satisfy or override those gates.

See [Unitree hardware references](UNITREE_G1_HARDWARE_REFERENCES.md) for the
manufacturer's specification, manuals, and robot models used as starting evidence.
