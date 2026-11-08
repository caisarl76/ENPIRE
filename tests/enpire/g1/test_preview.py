"""Preview mapping must not confuse recorded order with a model's traversal order."""

import numpy as np
import pytest

from enpire.env.forge.g1 import preview


def test_qpos_mapping_uses_joint_names_and_preserves_unmeasured_base_reference():
    # The model visits thumb before index; the recording stores index first.
    model_names = ["right_hand_thumb_0_joint", "right_hand_index_0_joint"]
    qpos = preview.map_qpos(
        ["right_hand_index_0_joint", "right_hand_thumb_0_joint"], [[0.4, -0.2]],
        model_names, [7, 8], [0, 0, 0.8, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(qpos, [[0, 0, 0.8, 1, 0, 0, 0, -0.2, 0.4]])


@pytest.mark.parametrize("names,model_names,addresses", [
    (["missing"], ["index"], [7]),
    (["index", "index"], ["index", "thumb"], [7, 8]),
    (["index"], ["index", "thumb"], [7, 8]),
    (["index", "thumb"], ["index", "thumb"], [7, 7]),
])
def test_missing_or_ambiguous_mapping_fails_instead_of_filling_fake_joints(names, model_names, addresses):
    with pytest.raises(ValueError):
        preview.map_qpos(names, np.zeros((1, len(names))), model_names, addresses, np.zeros(9))


def test_limit_report_flags_raw_values_without_clipping_them():
    state = np.array([[0.1, 0.0], [0.2, 0.4], [-0.3, 0.0]])
    original = state.copy()
    report = preview.limit_report(["elbow", "index"], state, [-0.2, 0], [0.2, 0.3])
    assert report["frames_outside_model_limits"] == 2
    assert report["joints"][0]["frames_outside"] == 1
    assert report["joints"][1]["frames_outside"] == 1
    assert report["joints"][0]["max_excess_rad"] == pytest.approx(0.1)
    np.testing.assert_array_equal(state, original)


def write_bundle(path, **updates):
    eef = np.zeros((3, 14))
    eef[:, [3, 10]] = 1
    values = dict(schema_version=1, fps=50, joint_names=[f"joint{i}" for i in range(43)],
                  state=np.zeros((3, 43)), eef_state=eef,
                  root_orientation=[[1, 0, 0, 0]] * 3,
                  episode_index=[0, 0, 0], frame_index=[0, 1, 2],
                  timestamp=[0, 0.02, 0.04], stream_mode=[1, 1, 2], video_paths=[""])
    values.update(updates)
    np.savez(path, **values)


@pytest.mark.parametrize("damage", [
    {"root_orientation": [[0, 0, 0, 0]] * 3},
    {"frame_index": [0, 1, 1]},
    {"episode_index": [0, 1, 0]},
    {"timestamp": [0, 0.02, 0.01]},
    {"state": np.full((3, 43), float("nan"))},
    {"fps": 0},
])
def test_preview_bundle_rejects_corruption_before_loading_a_model(tmp_path, damage):
    path = tmp_path / "bad.npz"
    write_bundle(path, **damage)
    with pytest.raises(ValueError):
        preview.load_bundle(path)


def test_preview_bundle_keeps_source_timestamps_and_frame_identity(tmp_path):
    path = tmp_path / "good.npz"
    write_bundle(path)
    bundle = preview.load_bundle(path)
    assert bundle["frame_index"].tolist() == [0, 1, 2]
    assert bundle["timestamp"].tolist() == [0, 0.02, 0.04]
    assert bundle["root_orientation"][2].tolist() == [1, 0, 0, 0]


def test_missing_schema_is_a_validation_error(tmp_path):
    path = tmp_path / "no-schema.npz"
    write_bundle(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files if key != "schema_version"}
    np.savez(path, **values)
    with pytest.raises(ValueError, match="schema"):
        preview.load_bundle(path)
