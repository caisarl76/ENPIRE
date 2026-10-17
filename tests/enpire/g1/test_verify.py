"""Proprioceptive trial verdicts must follow the right Dex3 hand, independent of the operator's verdict."""

import json

import numpy as np
import pytest

from enpire.cli import main
from enpire.env.forge.g1.verify import verify_session, verify_trial


def write_trial(path, closures, dt=0.02, lead=10):
    """A LeRobot streamer trial log: `lead` pre-episode rows, then one episode row per closure value."""
    closures = np.repeat(np.asarray(closures, np.float32), 25)  # each value held for 0.5 s
    n = lead + len(closures)
    state = np.zeros((n, 28), np.float32)
    state[lead:, 21:28] = (closures / 7)[:, None]
    phase = np.array(["wait first chunk"] * lead + ["episode"] * len(closures))
    t_ep = np.r_[np.full(lead, -1.0), dt * np.arange(len(closures))]
    np.savez(path, state=state, phase=phase, t_ep=t_ep)


@pytest.mark.parametrize(
    ("closures", "label"),
    [
        ([0.1, 0.1, 0.1], "no_grasp"),
        ([0.1, 4.0, 4.2, 4.2, 0.1, 0.1], "released"),
        ([0.1, 4.0, 4.2, 4.2, 4.2], "held_not_released"),
        ([0.1, 6.4, 6.4, 0.1], "missed"),
        ([0.1, 3.6, 5.1, 6.4, 0.9], "missed"),  # a partial hold that slipped to air
        ([0.1, 6.4, 2.7, 4.2, 4.2, 0.1], "missed"),  # regrasp after a miss: the first attempt failed
        ([0.1, 4.0, 0.1, 0.1, 4.0, 4.2, 4.2, 0.1], "released"),  # a 0.5 s flicker is not an attempt
    ],
)
def test_first_attempt_label(tmp_path, closures, label):
    write_trial(tmp_path / "trial_000.npz", closures)
    result = verify_trial(tmp_path / "trial_000.npz")
    assert result["label"] == label
    assert result["handover_confirmed"] is None  # a release alone does not prove the recipient took the object


def test_session_report_compares_with_operator_verdicts(tmp_path):
    write_trial(tmp_path / "trial_000.npz", [0.1, 4.0, 4.2, 4.2, 0.1])
    write_trial(tmp_path / "trial_001.npz", [0.1, 6.4, 6.4, 0.1])
    write_trial(tmp_path / "trial_002.npz", [0.1, 4.0, 4.2, 4.2, 0.1])
    verdicts = [("success", ""), ("fail", "grasp failed / missed"), ("fail", "dropped the object")]
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps({"trial": i, "verdict": v, "reason": r}) + "\n" for i, (v, r) in enumerate(verdicts))
    )
    report = verify_session(tmp_path)
    assert [t["label"] for t in report["trials"]] == ["released", "missed", "released"]
    assert report["operator_success_released"] == "1/1"
    assert report["operator_fail_not_released"] == "1/2"
    assert report["motion_enabled"] is False


def test_cli_writes_the_report(tmp_path, capsys):
    write_trial(tmp_path / "trial_000.npz", [0.1, 4.0, 4.2, 4.2, 0.1])
    (tmp_path / "results.jsonl").write_text(json.dumps({"trial": 0, "verdict": "success", "reason": ""}) + "\n")
    assert main(["g1", "verify-session", "--session", str(tmp_path), "--output", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["trials"][0]["label"] == "released"


def test_rejects_a_log_without_the_28d_state(tmp_path):
    np.savez(tmp_path / "trial_000.npz", state=np.zeros((5, 29)), phase=np.array(["episode"] * 5), t_ep=np.zeros(5))
    with pytest.raises(ValueError, match="28D"):
        verify_trial(tmp_path / "trial_000.npz")
