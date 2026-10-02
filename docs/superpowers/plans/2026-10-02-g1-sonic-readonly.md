# G1 SONIC observation implementation plan

> **For agentic workers:** Use the native inline execution workflow for these tasks, with one independent review at completion. Steps use checkbox syntax.

**Goal:** Inspect G1 camera/telemetry and bottle-handover demonstrations without motion.

**Architecture:** A lazy G1 facade consumes existing SONIC streams. External station
profiles and a conservative preflight report keep hardware evidence separate from
motion authorization. Offline dataset validation provides the initial task contract.

**Tech Stack:** Python 3.11, NumPy, MessagePack, ZeroMQ, OpenCV; optional PyArrow.

**Spec:** ../specs/2026-10-02-g1-sonic-readonly-design.md

## Global constraints

- No publisher, motor API, controller launch, or reset motion.
- No workstation paths, addresses, identifiers, or calibration values in Git.
- Dependencies remain optional and lazy; preserve existing environments.
- Never synthesize Dex3 joints for an absent or Inspire hand.
- Observations and dataset checks cannot claim physical task success.

## Review focus

Malformed timestamps; wrong body/hand order; absent telemetry; malformed external
profiles; missing or corrupt dataset episodes must fail with actionable errors.

## Task 1: Subscription tools and preflight

Files: `enpire/env/forge/g1/{__init__,station,transport,commands}.py`,
`enpire/cli.py`, `pyproject.toml`, `tests/enpire/g1/test_observation.py`.

Interfaces: `decode_camera(raw)`, `decode_state(raw, hand=...)`,
`Subscriber(endpoint, stream, hand=...).read_many(samples, timeout)`,
`load_station(path)`, `preflight(station, observations)`, `add_g1_parser(commands)`.

- [ ] Write and run failing tests for mapping, malformed input, lazy CLI imports,
  stalled streams, external-profile validation, and motion-disabled preflight.
- [ ] Implement bounded subscription-only tools and CLI init-station/observe/preflight.
- [ ] Run G1 and existing core CLI/foundation tests; lint changed Python files.

## Task 2: Offline task evidence and operating instructions

Files: `enpire/env/forge/g1/dataset.py`, `tests/enpire/g1/test_dataset.py`,
`enpire/env/docs/G1_SONIC_WORKFLOWS.md`, provenance, root lock.

Interface: `inspect_dataset(root: Path) -> dict` with counts, mode counts, finite
feature checks, joint-order validation, and descriptive bilateral ranges.

- [ ] Write and run failing tests for hand interleaving, missing/corrupt episodes,
  nonfinite features, and frame-count mismatch.
- [ ] Implement the reader, optional extras, CLI entry point, and workflow docs.
- [ ] Inspect the provided dataset and camera streams on both robots; save all
  station files and reports outside Git and label telemetry unverified if absent.
- [ ] Independently review the branch, fix material findings with regression tests,
  run required checks, and report remaining physical gates accurately.
