import importlib.util
import os
import tempfile
import unittest

import numpy as np


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SCRIPT_PATH = os.path.join(SERVER_PROJECT_ROOT, "scripts", "dagger_relabel.py")


class DaggerRelabelTests(unittest.TestCase):
    def test_output_reuses_payload_without_copying_old_teacher_labels(self):
        self.assertTrue(os.path.exists(SCRIPT_PATH), "dagger_relabel.py must exist")
        spec = importlib.util.spec_from_file_location(
            "dagger_relabel_under_test", SCRIPT_PATH
        )
        dagger_relabel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dagger_relabel)
        with tempfile.TemporaryDirectory() as tempdir:
            source = os.path.join(tempdir, "source")
            output = os.path.join(tempdir, "output")
            os.makedirs(os.path.join(source, "frames"))
            with open(os.path.join(source, "frames.json"), "w", encoding="utf-8") as f:
                f.write("[]")
            with open(os.path.join(source, "frames", "radar.npy"), "wb") as f:
                f.write(b"radar")
            np.save(os.path.join(source, "teacher_waypoints.npy"), np.ones((1, 5, 3)))
            np.save(os.path.join(source, "teacher_waypoint_valid.npy"), np.ones((1, 5)))

            dagger_relabel._link_sample_payload(source, output)

            self.assertTrue(os.path.islink(os.path.join(output, "frames.json")))
            self.assertTrue(os.path.islink(os.path.join(output, "frames")))
            self.assertFalse(os.path.exists(os.path.join(output, "teacher_waypoints.npy")))
            self.assertFalse(
                os.path.exists(os.path.join(output, "teacher_waypoint_valid.npy"))
            )


if __name__ == "__main__":
    unittest.main()
