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

```bash
enpire g1 observe --station "$STATION_JSON" --camera-only --output "$CAMERA_REPORT"
enpire g1 observe --station "$STATION_JSON" --output "$OBSERVATION_REPORT"
enpire g1 preflight --station "$STATION_JSON" --output "$PREFLIGHT_REPORT"
```

The default is five advancing samples with a 15-second total deadline per stream.
Missing, malformed, or stalled streams fail. `observe --camera-only` can pass
while SONIC is offline. Full observation requires its existing `g1_debug` state
publisher; the tool never starts it. Camera reports describe decoded RGB images.
Dex3 state interleaves the left hand before the right arm to match the dataset's
43-joint layout. Other hand types report the measured 29-joint body only.
No unmeasured hand or end-effector pose is filled with artificial zeros.

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
