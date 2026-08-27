import json
import os
import sys
import tempfile
import unittest

import numpy as np


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SERVER_SCRIPTS = os.path.join(SERVER_PROJECT_ROOT, "scripts")
if SERVER_SCRIPTS not in sys.path:
    sys.path.insert(0, SERVER_SCRIPTS)

import build_spl_reference_paths


class BuildSplReferencePathsTests(unittest.TestCase):
    def test_collision_validation_uses_authoritative_clearance_radius(self):
        labels = {"clearance": np.asarray([0.31, 0.16, 0.40], dtype=np.float32)}

        minimum = build_spl_reference_paths.validate_collision_clearance(
            labels, collision_radius=0.15
        )

        self.assertAlmostEqual(minimum, 0.16, places=6)
        with self.assertRaisesRegex(RuntimeError, "collision radius"):
            build_spl_reference_paths.validate_collision_clearance(
                {"clearance": np.asarray([0.30, 0.14], dtype=np.float32)},
                collision_radius=0.15,
            )

    def test_plan_until_endpoint_replans_partial_fast_planner_paths(self):
        class PartialPlanner:
            def __init__(self):
                self.calls = 0

            def plan_waypoints(self, route, start_vel=None, start_acc=None):
                self.calls += 1
                start = np.asarray(route[0], dtype=np.float32)
                goal = np.asarray(route[-1], dtype=np.float32)
                if np.linalg.norm(goal - start) <= 0.25:
                    end = goal
                else:
                    end = start + 0.5 * (goal - start)
                return [start, end]

        class Labeler:
            def __init__(self):
                self.planner = PartialPlanner()

            def _route_for(self, position):
                return [tuple(position), (1.0, 0.0, 0.0)]

        labeler = Labeler()
        path, replans = build_spl_reference_paths.plan_until_endpoint(
            labeler,
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            timeout_s=1.0,
            max_replans=10,
        )

        self.assertGreater(labeler.planner.calls, 1)
        self.assertEqual(replans, labeler.planner.calls)
        np.testing.assert_allclose(path[-1], [1.0, 0.0, 0.0], atol=1e-6)

    def test_plan_until_endpoint_reaches_semantic_guide_before_goal(self):
        class Planner:
            def __init__(self):
                self.targets = []

            def plan_waypoints(self, route, start_vel=None, start_acc=None):
                start = np.asarray(route[0], dtype=np.float32)
                target = np.asarray(route[-1], dtype=np.float32)
                self.targets.append(target.copy())
                return [start, target]

        class Labeler:
            def __init__(self):
                self.planner = Planner()

            def _route_for(self, position):
                return [tuple(position), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]

        labeler = Labeler()
        path, _ = build_spl_reference_paths.plan_until_endpoint(
            labeler,
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            timeout_s=1.0,
        )

        np.testing.assert_allclose(labeler.planner.targets[0], [0.0, 1.0, 0.0])
        self.assertTrue(np.any(np.all(np.isclose(path, [0.0, 1.0, 0.0]), axis=1)))
        np.testing.assert_allclose(path[-1], [1.0, 1.0, 0.0])

    def test_path_length_uses_three_dimensional_arc_length(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 2.0]],
            dtype=np.float32,
        )

        length = build_spl_reference_paths.path_length_3d(points)

        self.assertAlmostEqual(length, 7.0)

    def test_truncate_at_first_completion_keeps_completion_point(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        completion = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        truncated, index = build_spl_reference_paths.truncate_at_first_completion(
            points, completion
        )

        self.assertEqual(index, 2)
        np.testing.assert_array_equal(truncated, points[:3])

    def test_truncate_rejects_trajectory_without_completion(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "never reaches completion"):
            build_spl_reference_paths.truncate_at_first_completion(
                points, np.zeros((2,), dtype=np.float32)
            )

    def test_save_reference_writes_readable_npz_json_and_metadata(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32
        )
        metadata = {
            "status": "valid",
            "skill": "through",
            "scene_id": "scene_001",
            "reference_path_length_m": 3.0,
        }
        with tempfile.TemporaryDirectory() as output_dir:
            sample_output = build_spl_reference_paths.save_reference(
                output_dir, metadata, points
            )

            with np.load(os.path.join(sample_output, "reference_path.npz")) as archive:
                np.testing.assert_array_equal(archive["points_std"], points)
            with open(os.path.join(sample_output, "trajectory.json"), encoding="utf-8") as f:
                trajectory = json.load(f)
            with open(os.path.join(sample_output, "metadata.json"), encoding="utf-8") as f:
                saved_metadata = json.load(f)
            self.assertEqual(trajectory["points_std"], points.tolist())
            self.assertEqual(saved_metadata, metadata)

    def test_load_manifest_index_rejects_duplicate_scene_ids(self):
        sample = {
            "skill": "over",
            "scene_id": "scene_001",
            "relative_path": "over/scene_001",
            "category": "box",
        }
        with tempfile.TemporaryDirectory() as directory:
            for batch_id in (1, 2):
                with open(
                    os.path.join(directory, f"batch_{batch_id}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"samples": [sample]}, f)
            with open(
                os.path.join(directory, "manifest_index.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "data_root": "/dataset",
                        "manifests": [
                            {"file": "batch_1.json"},
                            {"file": "batch_2.json"},
                        ],
                    },
                    f,
                )

            with self.assertRaisesRegex(ValueError, "duplicate manifest sample"):
                build_spl_reference_paths.load_manifest_index(
                    os.path.join(directory, "manifest_index.json")
                )


if __name__ == "__main__":
    unittest.main()
