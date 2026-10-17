"""Offline, proprioceptive verdicts for recorded G1 bottle-handover trials.

Reads LeRobot ``sonic_policy_streamer.py --trials`` session logs (``trial_{i:03d}.npz``: 28D measured state per 50 Hz
tick, phase, t_ep) and labels each trial's first grasp attempt from the right Dex3 closure alone, independently of
the operator. The right hand grasps; its closure is the sum of |q| over its 7 joints (rad). Robot sessions s01-s08
(2026-10-08..16, one drink bottle): open ~0.1, closed on the bottle 3.6-4.4 (5.1 once, a partial hold), closed on air
6.0-6.5.

Labels: ``no_grasp`` (never closed for an attempt), ``missed`` (the first attempt closed on air, possibly after a
partial hold), ``held_not_released`` (still holding when the trial ended) and ``released`` (held, then opened).
A release at the presentation pose looks the same as dropping the bottle there, so ``released`` is only a candidate
success: ``handover_confirmed`` stays None until recipient acquisition is checked independently (e.g. camera).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RIGHT_HAND = slice(21, 28)  # 28D state: arms 0-13, left Dex3 14-20, right Dex3 21-27
CLOSED = 3.0  # an attempt starts above this
AIR = 5.5  # closed on air above this
OPEN = 1.0  # an attempt ends (released) below this
MIN_ATTEMPT_S = 1.0  # shorter closures that never reach AIR are flickers, not attempts


def _attempts(closure: np.ndarray, t: np.ndarray) -> list[dict]:
    out, i, n = [], 0, len(closure)
    while i < n:
        if closure[i] <= CLOSED:
            i += 1
            continue
        j = i
        while j < n and not closure[j] < OPEN:  # NaN (no state) keeps the attempt open
            j += 1
        seg = closure[i:j]
        out.append({"start_s": float(t[i]), "end_s": float(t[j - 1]), "released": j < n,
                    "closed_on_air": bool(np.nanmax(seg) > AIR)})
        i = j
    return [a for a in out if a["closed_on_air"] or a["end_s"] - a["start_s"] >= MIN_ATTEMPT_S]


def verify_trial(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        state, phase, t_ep = z["state"], z["phase"], z["t_ep"]
    if state.ndim != 2 or state.shape[1] != 28:
        raise ValueError(f"{path}: expected the 28D streamer state, got {state.shape}")
    episode = phase == "episode"
    closure = np.abs(state[episode, RIGHT_HAND]).sum(1)
    attempts = _attempts(closure, t_ep[episode])
    if not attempts:
        label = "no_grasp"
    elif attempts[0]["closed_on_air"]:
        label = "missed"
    else:
        label = "released" if attempts[0]["released"] else "held_not_released"
    return {"trial": Path(path).name, "label": label, "attempts": attempts, "handover_confirmed": None}


def verify_session(session: Path) -> dict:
    """Label every trial in results.jsonl and compare with the operator's verdicts (not used for the labels)."""
    session = Path(session)
    rows = [json.loads(line) for line in (session / "results.jsonl").read_text().splitlines()]
    trials = []
    for row in rows:
        result = verify_trial(session / f"trial_{row['trial']:03d}.npz")
        trials.append({**result, "operator_verdict": row["verdict"], "operator_reason": row.get("reason", "")})
    ok = [t for t in trials if t["operator_verdict"] == "success"]
    bad = [t for t in trials if t["operator_verdict"] != "success"]
    return {
        "session": str(session.resolve()),
        "trials": trials,
        # every operator success needs a released first attempt; operator fails labelled released are drops (or
        # regrasps) that only a recipient check can separate from a handover
        "operator_success_released": f"{sum(t['label'] == 'released' for t in ok)}/{len(ok)}",
        "operator_fail_not_released": f"{sum(t['label'] != 'released' for t in bad)}/{len(bad)}",
        "thresholds": {"closed": CLOSED, "air": AIR, "open": OPEN, "min_attempt_s": MIN_ATTEMPT_S},
        "motion_enabled": False,
    }
