"""Run directly in an existing MuJoCo environment, or through pytest when installed."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "Use the existing MuJoCo environment for this model check")
class PreviewModelTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location("g1_preview", root / "enpire/env/forge/g1/preview.py")
        self.preview = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.preview)
        self.model = mujoco.MjModel.from_xml_string('''
            <mujoco><compiler angle="radian"/><worldbody><body name="pelvis">
            <freejoint/><geom size="0.05"/>
            <body name="left_wrist_yaw_link" pos="0.2 0.3 0.4">
              <joint name="left_elbow" axis="0 0 1" range="-0.2 0.2"/><geom size="0.05"/>
            </body><body name="right_wrist_yaw_link" pos="0.5 -0.3 0.4">
              <joint name="right_elbow" axis="0 0 1" range="-0.2 0.2"/><geom size="0.05"/>
            </body></body></worldbody></mujoco>''')
        eef = np.tile([0.2, 0.3, 0.4, 1, 0, 0, 0, 0.5, -0.3, 0.4, 1, 0, 0, 0], (3, 1))
        self.bundle = dict(joint_names=["right_elbow", "left_elbow"], state=np.zeros((3, 2)),
                           eef_state=eef, episode_index=np.zeros(3, dtype=int), fps=50)

    def test_inspection_matches_named_fk_and_never_claims_physical_readiness(self):
        report = self.preview.inspect_poses(self.bundle, self.model)
        self.assertLess(report["eef_fk"]["right"]["max_position_error_m"], 1e-10)
        self.assertEqual(report["limits"]["frames_outside_model_limits"], 0)
        self.assertFalse(report["motion_enabled"])
        self.assertFalse(report["root_translation_measured"])
        self.assertFalse(report["balance_verified"])

    def test_out_of_limit_pose_is_reported_without_clipping(self):
        self.bundle["state"][1, 0] = 0.4
        report = self.preview.inspect_poses(self.bundle, self.model)
        self.assertEqual(report["limits"]["frames_outside_model_limits"], 1)
        self.assertAlmostEqual(report["eef_fk"]["right"]["max_rotation_error_deg"], 22.918311805, places=6)
        self.assertEqual(self.bundle["state"][1, 0], 0.4)

    def test_render_keeps_rgb_frame_identity_and_recording_speed(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 50, (640, 480))
            self.assertTrue(writer.isOpened())
            for blue in (50, 80, 110):
                writer.write(np.full((480, 640, 3), [blue, 100, 150], dtype=np.uint8))
            writer.release()
            self.bundle.update(video_paths=[str(source)], frame_index=np.arange(3),
                               timestamp=np.array([0, 0.02, 0.04]),
                               root_orientation=np.tile([1, 0, 0, 0], (3, 1)))
            output = Path(directory) / "preview.avi"
            report = self.preview.render_preview(self.bundle, self.model, 0, output, stride=2)
            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["fps"], 25)
            reader = cv2.VideoCapture(str(output))
            self.assertEqual(reader.get(cv2.CAP_PROP_FRAME_COUNT), 2)
            for expected_blue in (50, 110):
                ok, image = reader.read()
                self.assertTrue(ok)
                self.assertLess(abs(int(image[200, 200, 0]) - expected_blue), 6)
            reader.release()
            # A direct caller must also preserve a pre-existing companion PNG.
            output.unlink()
            sheet = output.with_suffix(".png")
            sheet.write_bytes(b"existing contact sheet")
            with self.assertRaises(FileExistsError):
                self.preview.render_preview(self.bundle, self.model, 0, output, stride=2)
            self.assertEqual(sheet.read_bytes(), b"existing contact sheet")


if __name__ == "__main__":
    unittest.main()
