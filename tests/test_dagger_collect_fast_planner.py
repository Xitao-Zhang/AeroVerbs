import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SERVER_SCRIPTS = os.path.join(SERVER_PROJECT_ROOT, "scripts")
if SERVER_SCRIPTS not in sys.path:
    sys.path.insert(0, SERVER_SCRIPTS)

import dagger_collect


class DaggerCollectFastPlannerTests(unittest.TestCase):
    def test_fast_planner_and_collision_defaults(self):
        argv = [
            "dagger_collect.py",
            "--checkpoint",
            "student.pt",
            "--output-dir",
            "dagger_out",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = dagger_collect.parse_args()

        self.assertAlmostEqual(args.drone_radius, 0.15)
        self.assertIsNone(args.teacher_preferred_clearance)
        self.assertEqual(args.sample_seed, 42)
        self.assertEqual(args.teacher_replan_interval, 5)
        self.assertAlmostEqual(args.teacher_timeout, 2.0)
        self.assertEqual(args.teacher_max_iter, 2000)

    def test_save_sample_persists_planner_diagnostics(self):
        with tempfile.TemporaryDirectory() as tempdir:
            heatmap_src = os.path.join(tempdir, "source_heatmap.npy")
            np.save(heatmap_src, np.zeros((2, 2), dtype=np.float32))
            sample_dir = os.path.join(tempdir, "sample")
            stats = {
                "teacher": "fast_planner_online",
                "sdf_source": "precomputed_npz",
                "replan_attempts": 2,
                "fallback_frames": 1,
            }

            dagger_collect._save_sample(
                sample_dir=sample_dir,
                frames=[],
                raw_ranges=[],
                downward_ranges=[],
                teacher_wp=[],
                teacher_valid=[],
                teacher_comp=[],
                teacher_sources=["invalid", "fast_planner_cache"],
                teacher_weights=[0.0, 1.0],
                meta={"skill": "around_left"},
                heatmap_src=heatmap_src,
                planner_stats=stats,
            )

            with open(
                os.path.join(sample_dir, "teacher_planner_stats.json"), encoding="utf-8"
            ) as f:
                saved = json.load(f)
            self.assertEqual(saved, stats)
            np.testing.assert_array_equal(
                np.load(os.path.join(sample_dir, "teacher_label_source.npy")),
                ["invalid", "fast_planner_cache"],
            )
            np.testing.assert_array_equal(
                np.load(os.path.join(sample_dir, "teacher_frame_weight.npy")),
                [0.0, 1.0],
            )

    def test_teacher_completion_terminates_collection(self):
        self.assertTrue(dagger_collect._teacher_completion_is_terminal(1))
        self.assertFalse(dagger_collect._teacher_completion_is_terminal(0))


if __name__ == "__main__":
    unittest.main()
