# G1 Offline Adapter Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation and a fresh final reviewer. Steps use checkbox syntax for tracking.

**Goal:** Define the bottle-task policy boundary and qualify a SONIC motion-token adapter without robot connections.

**Architecture:** A pure adapter gates the existing v4 byte encoder behind fresh camera/state samples and an exclusive in-memory writer lease. An offline runner supplies recorded POSE actor outputs and synthetic receipt clocks; all physical flags remain false.

**Tech Stack:** Existing Python 3.11 ENPIRE environment, standard library, lazy NumPy/PyArrow, pinned upstream SONIC encoder.

**Spec:** docs/superpowers/specs/2026-10-06-g1-offline-adapter-design.md

## Global constraints

- No network transport, robot SSH, controller launch, or new dependencies.
- No workstation identifiers or paths embedded in committed code.
- Load the upstream encoder explicitly; encode token-only packets as v4.
- All reports leave physical success and hardware motion disabled.
- Never treat stopping local output as a verified receiver fallback.

## Review focus

- A duplicate source sample must not renew liveness.
- A delayed encoder must not commit after data or ownership expires.
- A renewed/taken-over lease must invalidate previous generations.
- Recent actor creation must not hide stale policy input.
- A mode boundary must end the reference session before another run starts.

## Task 1: Policy boundary and adapter

**Files:** Create enpire/env/forge/g1/adapter.py and tests/enpire/g1/test_adapter.py.

**Interfaces:** PoseTarget declares token, hands, hand order, frame index, creation time, and input receipt time. MemoryBus acquires/releases/validates leases. OfflineAdapter exposes observe_state, observe_camera, start, submit, tick, stop, and report with injected clock and encoder.

- [x] Write behavioral tests for literal packet fields/order and competing writers, freshness, encoder delay, idle stop, and latched faults.
- [x] Run tests and observe feature-missing failure.
- [x] Implement the smallest pure adapter satisfying those contracts.
- [x] Run targeted tests, G1/core checks, and Ruff.

## Task 2: Dataset qualification and CLI

**Files:** Create enpire/env/forge/g1/shadow.py and tests/enpire/g1/test_shadow.py; modify g1/commands.py, G1_SONIC_WORKFLOWS.md, and source_provenance.yaml.

**Interfaces:** qualify_dataset(dataset: Path, encoder_path: Path) -> dict, exported through g1 shadow-adapter with a required new output JSON. The runner creates no publisher and records summaries only.

- [x] Write tests for v4 qualification, mode/episode boundaries, socket prohibition, and preservation of existing report files.
- [x] Observe failure, implement runner/CLI, and run tests to green.
- [x] Qualify the real dataset using the explicit upstream encoder; save external evidence.
- [x] Update workflow/provenance, obtain final source review, fix important findings with regression tests, and run project checks.
- [x] Commit local source and update external status. No push or hardware rollout.

## Verification and review record

- Adapter and runner tests failed on missing modules before implementation, then
  passed. Float32 overflow rejection was tightened after a warnings-as-errors
  regression reproduced the casting warning.
- The first real corpus qualification passed 18,831 POSE records across 43
  sessions, excluding 139 other-mode records. No socket attempts or saved/sent
  packets occurred.
- Fresh independent gpt-6-astra/high review found overlapping output submissions
  could reorder frames. A regression reproduced the bug; submissions now reject
  overlap and preserve the original lease generation through encoding.
- Review's expired-lease clock reversal and clock-provider exception findings
  were treated as correctness issues in the advertised lease/fault contract.
  Both regressions failed before fixes, then passed. Stop during encoding also
  has a characterization check. No deferred findings remain.
- Physical policy behavior, receiver fallback, and real cross-process ownership
  remain outside this offline implementation and are explicitly unverified.
- The full ENPIRE suite still has the six existing optional-dependency
  collection errors documented in the external status report. No unrelated
  dependency installation was attempted.
- Final verification: 98 scoped G1/core checks passed, with three MuJoCo skips
  in the ENPIRE environment; Ruff passed. Final real corpus qualification again
  passed 18,831 POSE samples in 43 sessions with no socket attempts or saved/sent
  packets.
