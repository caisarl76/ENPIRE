# G1 SONIC observation and bottle-handover preparation

The first task is right-hand drink-bottle pickup from a black box, followed by
palm-up presentation with spread fingers for a human receiver. GEAR-SONIC owns
balance, locomotion, teleoperation, and all existing control safeguards. ENPIRE
must first make observations and task evidence inspectable on both robots.

This change adds subscription-only camera and state tools and an offline dataset
inspection command. It has no controller launcher, publisher, reset motion, or
motor API. External station JSON contains robot identity, hand type, stream
endpoints, and references to calibration and operator-preflight records. No
station-specific paths or addresses enter version control. Imports stay lazy.

Camera messages use the existing composed_camera MessagePack/JPEG protocol.
State messages use the existing `g1_debug` topic prefix plus MessagePack protocol.
Validation rejects missing fields, wrong joint lengths, nonfinite values, and
invalid quaternions. Multiple samples must advance their source timestamps or
state index within a bounded deadline. Receive times use the local monotonic
clock; cross-host clock synchronization is not inferred from receipt liveness.

For Dex3 only, measured MuJoCo body joints (29) and hand joints (7 each) map to
the dataset's 43-joint order: body[0:22], left hand, body[22:29], right hand.
An absent or Inspire hand must never masquerade as Dex3 state. Images decode
to RGB. No end-effector pose is invented without matching kinematics.

Offline inspection reads LeRobot v2.1 metadata and Parquet without writing the
dataset. It validates dimensions, finite state/action/hand/token values,
episode/frame counts, and documented joint ordering. It reports stream-mode
counts and bilateral movement ranges as descriptive evidence, not success.
The recorded full-body action and 18-value VR rotation representation are never
forwarded to SONIC. No dataset playback is exposed by this change.

Preflight reports separate stream readiness from physical readiness. Calibration
and safety record existence alone cannot authorize motion. Motion readiness
always remains false while the actuation bridge, independent task verifier,
calibration validation, and explicit operator authorization are unfinished.
The head camera's current RGB stream does not establish depth or a calibrated
surface plane. Physical verification must distinguish presentation from actual
recipient acquisition before releasing the bottle.

Verification uses pure fixtures for decoder errors and joint ordering, local
ZeroMQ publishers for bounded subscriptions and stalled streams, CLI tests,
the real offline dataset, and actual camera streams. Real state validation waits
for an operator-controlled SONIC telemetry session; mock results remain labeled.
The factory G1 29-joint model fixes the head to the torso. Calibration must
therefore validate the camera mounting frame and the torso/waist chain rather
than assume a separately actuated neck. Hardware revision matching remains a
physical preflight requirement; G1+ models have a different head chain.
