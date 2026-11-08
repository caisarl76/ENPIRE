# G1 bottle-task controller and offline SONIC adapter

## Outcome and scope

Prepare the first right-Dex3 bottle pickup and stable palm-up presentation task.
GEAR-SONIC remains the body/hand control owner. This change defines the policy
boundary and implements a socket-free adapter qualification harness. It cannot
start a controller, change a station gate, or enable hardware motion.

## Controller choice

Use an ENPIRE staged task supervisor around a motion-token actor. The stages are
localize, approach, grasp, lift, and supported palm-up presentation. Independent
measurements must authorize transitions; a final pose or nearby human hand
does not prove bottle support or acquisition. Recipient acquisition is a
separate optional outcome. No automatic release or reset is selected here.

The actor receives fresh RGB and measured G1/Dex3 state plus the supervisor's
phase/goal, and returns one 64-value SONIC motion token and two seven-value Dex3
hand targets. The adapter requires declared hand order and translates
index/middle/thumb policy output to thumb/index/middle motor order. Already
native teleoperation hand targets are preserved. SONIC performs low-level
whole-body execution. Motion tokens include whole-body intent; they are not
an arm-only layer that can be mixed with an independent teleoperation writer.

This is the selected execution interface, not a trained bottle policy. The
current reference actor is saved POSE samples for offline qualification. A
compatible policy/checkpoint or characterized pose-to-token encoder, calibrated
goals, and independent stage evidence are still needed for autonomous picking.
Raw 43-value action.wbc and stored EEF coordinates are not converted to commands.

## Source contract

Pinned GR00T source: 7c6d710b783a984ede504d24a65bc70e486b3b1d.
The existing pack_pose_message builder is loaded explicitly from a local file;
its byte encoder is reused without copying the control algorithm.

The receiver's DecodeIntoMotionSequence token-only path requires protocol v4.
The packet topic is pose, its JSON header is 1,280 bytes, and its fields are:

| Field | Type and shape | Meaning |
|---|---|---|
| token_state | float32 [1,64] | One whole-body motion token |
| frame_index | int64 [1] | Strictly increasing within an adapter session |
| left_hand_joints | float32 [1,7] | Native Dex3 thumb/index/middle order |
| right_hand_joints | float32 [1,7] | Native Dex3 thumb/index/middle order |

Version 3 requires body quaternion, SMPL pose/joints, and joint position/velocity;
it cannot carry the selected token-only contract. The earlier v3 experiment was
an encoder round-trip check, not receiver compatibility evidence.

## Adapter and ownership

Implement OfflineAdapter with an explicit upstream encoder, shared MemoryBus,
and injected monotonic clock. MemoryBus stores packets only in memory and
models exclusive writer leases with unique generations. Expiration invalidates
old handles; teleoperation takeover revokes the task lease. The bus validates
the lease again when committing a packet and rejects decreasing caller times.
Expiration remains invalid once observed. No real publisher or endpoint is
accepted. This models local ownership only: real cross-process arbitration and
controller takeover acknowledgement remain unverified.

Require two advancing source samples for both camera and state before start.
Duplicate source timestamps/indices never refresh liveness. Source resets,
future receipt times, nonfinite values, stale inputs, stale actor outputs,
regressing frame indices, loss of ownership, and encoder errors latch a fault.
Targets carry creation and policy-input receipt times in the same local
monotonic clock domain; input age is checked separately from inference age.
Freshness is checked again after encoding to catch a blocked encoder.
Overlapping submissions latch a fault and cancel in-flight work. The original
lease generation and frame ordering are checked before committing. Lifecycle
updates/commit are synchronized, while encoding runs outside the lifecycle lock
so stop and watchdog calls can cancel a blocked encoder. Clock provider failures
also latch a fault.

The adapter's tick is its idle watchdog; the future runner must call it even
when no actor output arrives. The qualification runner exercises this explicitly.
An explicit stop or fault ends the session, releases its lease if still owned,
and rejects subsequent outputs. A new instance and explicit start are required
to resume. Stopping output does not erase receiver-held tokens, stop the robot,
release fingers, or prove a safe fallback. No emergency-stop packet is inferred.

Age and lease settings are positive finite software test parameters, not
validated physical limits. Hand operating bounds, rate/step limits, transition
ramps, receiver timeouts, and safe receiver fallback require later simulation
and hardware characterization. Every report keeps motion_enabled and
physical_success_verified false.

## Deliverables and acceptance

- Pure adapter/lease implementation, with no network transport or new packages.
- Lazy g1 shadow-adapter CLI accepting external dataset, encoder, and new report
  paths. Validate the existing dataset contract, qualify contiguous POSE runs,
  and explicitly stop/restart reference sessions at mode/episode boundaries.
- Replay all eligible samples through the real upstream v4 byte encoder, using
  synthetic local camera/state receipt times. Do not claim image/state alignment.
- Block socket creation during qualification. Never save command packets.
- Tests cover native and dataset hand order, exact v4 fields, freshness,
  duplicates/resets, invalid dimensions/nonfinite/float32 overflow, competing
  owners, expired generation, takeover, idle watchdog, stop/no restart, and
  encoder failure or delay. Existing G1/core checks must continue to pass.
- Independent review and a report of remaining physical/receiver requirements.

## Decision tradeoffs

Planner upper-body targets are an alternative, but would require a separate
validated IK/planning path and characterization of waist/arm conventions and
mode transitions. The recorded token contract directly matches the current
dataset's POSE actions. It trades that compatibility for whole-body policy
dependence and requires exclusive control of the selected SONIC input session.
Direct motor targets would duplicate SONIC's control responsibility and are
outside this adapter.
