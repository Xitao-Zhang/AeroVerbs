import json
import os
import tempfile
import unittest

import numpy as np

from aeroverbs.data.fast_planner_teacher import (
    FastPlannerTeacherConfig,
    FastPlannerTeacherLabeler,
)


class _FallbackTeacher:
    def __init__(self):
        self.label_calls = 0

    def label_completion(self, prefix_points, current_position=None):
        return 1

    def label(self, position, forward, prefix_points):
        self.label_calls += 1
        return (
            np.full((5, 3), 9.0, dtype=np.float32),
            np.ones((5,), dtype=np.float32),
            1,
        )


class _RecordingPlanner:
    def __init__(self, fail=False):
        self.fail = fail
        self.routes = []
        self.start_velocities = []

    def plan_waypoints(self, route, start_vel=None, start_acc=None):
        del start_acc
        self.routes.append(np.asarray(route, dtype=np.float32))
        self.start_velocities.append(
            None if start_vel is None else np.asarray(start_vel, dtype=np.float32)
        )
        if self.fail:
            raise RuntimeError("no path")
        start = np.asarray(route[0], dtype=np.float32)
        return [tuple(start + np.array([0.1 * i, 0.0, 0.0])) for i in range(31)]


class _GuideCrossingPlanner(_RecordingPlanner):
    def plan_waypoints(self, route, start_vel=None, start_acc=None):
        del start_acc
        self.routes.append(np.asarray(route, dtype=np.float32))
        self.start_velocities.append(
            None if start_vel is None else np.asarray(start_vel, dtype=np.float32)
        )
        start = np.asarray(route[0], dtype=np.float32)
        return [
            tuple(start + np.array([min(0.3 * i, 4.0), 0.0, 0.0]))
            for i in range(31)
        ]


class FastPlannerTeacherTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_root = self.tempdir.name
        self.sample_dir = os.path.join(self.data_root, "around_left", "scene_001")
        os.makedirs(self.sample_dir)
        with open(os.path.join(self.sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"skill": "around_left", "scene_id": "scene_001"}, f)
        frames = [
            {"camera_position_std": [0.0, 0.0, 1.0], "camera_forward_std": [1.0, 0.0, 0.0]},
            {"camera_position_std": [4.0, 0.0, 1.0], "camera_forward_std": [1.0, 0.0, 0.0]},
        ]
        with open(os.path.join(self.sample_dir, "frames.json"), "w", encoding="utf-8") as f:
            json.dump(frames, f)
        with open(
            os.path.join(self.sample_dir, "heatmap_gate_region.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"points": [[2.0, -0.2, 1.0], [2.0, 0.2, 1.0]]}, f)
        with open(
            os.path.join(self.sample_dir, "heatmap_gate_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"center_std": [2.0, 0.0, 1.0], "normal_std": [1.0, 0.0, 0.0]}, f)
        sdf_dir = os.path.join(self.data_root, "_sdf", "around_left")
        os.makedirs(sdf_dir)
        np.savez_compressed(
            os.path.join(sdf_dir, "scene_001.npz"),
            grid=np.full((4, 4, 4), 2.0, dtype=np.float32),
            origin=np.array([1.0, -1.0, 0.0], dtype=np.float32),
            voxel_size=np.float32(0.5),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _make_teacher(self, planner, replan_interval=5, timeout_s=10.0):
        fallback = _FallbackTeacher()
        teacher = FastPlannerTeacherLabeler(
            self.sample_dir,
            data_root=self.data_root,
            config=FastPlannerTeacherConfig(
                replan_interval=replan_interval,
                timeout_s=timeout_s,
            ),
            fallback_teacher=fallback,
            planner_factory=lambda scene, config, sdf_map: planner,
        )
        return teacher, fallback

    def test_reuses_plan_for_five_frames_and_emits_body_frame_offsets(self):
        planner = _RecordingPlanner()
        teacher, fallback = self._make_teacher(planner, replan_interval=5)

        outputs = []
        for i in range(5):
            position = np.array([0.1 * i, 0.0, 1.0], dtype=np.float32)
            outputs.append(teacher.label(position, np.array([1.0, 0.0, 0.0]), [position]))

        self.assertEqual(len(planner.routes), 1)
        np.testing.assert_allclose(planner.routes[0][1], [2.0, 0.0, 1.0])
        waypoints, valid, completion = outputs[0]
        np.testing.assert_allclose(waypoints[:, 0], [0.3, 0.6, 1.0, 1.5, 2.0], atol=1e-6)
        np.testing.assert_allclose(waypoints[:, 1:], 0.0, atol=1e-6)
        np.testing.assert_array_equal(valid, np.ones(5, dtype=np.float32))
        self.assertEqual(completion, 1)
        self.assertEqual(fallback.label_calls, 0)

    def test_drops_semantic_guide_after_crossing_its_plane(self):
        planner = _RecordingPlanner()
        teacher, _ = self._make_teacher(planner, replan_interval=1)
        position = np.array([3.0, 0.0, 1.0], dtype=np.float32)

        teacher.label(position, np.array([1.0, 0.0, 0.0]), [position])

        self.assertEqual(planner.routes[0].shape, (2, 3))
        np.testing.assert_allclose(planner.routes[0][-1], [4.0, 0.0, 1.0])

    def test_around_keeps_guide_when_close_but_not_across_its_plane(self):
        planner = _RecordingPlanner()
        teacher, _ = self._make_teacher(planner, replan_interval=1)
        position = np.array([1.9, 0.0, 1.0], dtype=np.float32)

        teacher.label(position, np.array([1.0, 0.0, 0.0]), [position])

        self.assertEqual(planner.routes[0].shape, (3, 3))
        np.testing.assert_allclose(planner.routes[0][1], teacher.guide)

    def test_replan_uses_velocity_from_the_latest_executed_step(self):
        planner = _RecordingPlanner()
        teacher, _ = self._make_teacher(planner, replan_interval=5)

        for i in range(6):
            position = np.array([0.1 * i, 0.0, 1.0], dtype=np.float32)
            teacher.label(position, np.array([1.0, 0.0, 0.0]), [position])

        self.assertEqual(len(planner.start_velocities), 2)
        np.testing.assert_allclose(planner.start_velocities[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(planner.start_velocities[1], [1.0, 0.0, 0.0], atol=1e-5)

    def test_default_map_keeps_lateral_recovery_room_around_expert_path(self):
        planner = _RecordingPlanner()
        teacher, _ = self._make_teacher(planner)

        self.assertTrue(teacher.sdf_map.is_in_map([0.0, 1.75, 1.0]))

    def test_default_preferred_clearance_depends_on_skill(self):
        expected_by_skill = {
            "around_left": 0.50,
            "around_right": 0.50,
            "over": 0.20,
            "through": 0.15,
            "under": 0.15,
        }
        source_sdf = os.path.join(self.data_root, "_sdf", "around_left", "scene_001.npz")
        source = np.load(source_sdf)
        for skill, expected in expected_by_skill.items():
            with self.subTest(skill=skill):
                with open(os.path.join(self.sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"skill": skill, "scene_id": "scene_001"}, f)
                sdf_dir = os.path.join(self.data_root, "_sdf", skill)
                os.makedirs(sdf_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(sdf_dir, "scene_001.npz"),
                    grid=source["grid"],
                    origin=source["origin"],
                    voxel_size=source["voxel_size"],
                )
                planner = _RecordingPlanner()
                teacher, _ = self._make_teacher(planner)

                self.assertAlmostEqual(teacher.config.collision_radius, 0.15)
                self.assertAlmostEqual(teacher.preferred_clearance, expected)
                self.assertAlmostEqual(teacher.planner_config.clearance, 0.15)
                self.assertAlmostEqual(
                    teacher.planner_config.optimization_dist0,
                    max(expected - 0.15, 0.0),
                )
                self.assertAlmostEqual(teacher.stats()["preferred_clearance_m"], expected)

    def test_explicit_preferred_clearance_overrides_skill_default(self):
        planner = _RecordingPlanner()
        fallback = _FallbackTeacher()
        teacher = FastPlannerTeacherLabeler(
            self.sample_dir,
            data_root=self.data_root,
            config=FastPlannerTeacherConfig(preferred_clearance=0.40),
            fallback_teacher=fallback,
            planner_factory=lambda scene, config, sdf_map: planner,
        )

        self.assertAlmostEqual(teacher.preferred_clearance, 0.40)
        self.assertAlmostEqual(teacher.planner_config.optimization_dist0, 0.25)

    def test_region_softly_biases_expert_crossing_without_using_the_centroid(self):
        with open(
            os.path.join(self.sample_dir, "heatmap_gate_region.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"points": [[2.0, 0.1, 1.0], [2.0, 0.8, 1.0]]}, f)
        planner = _RecordingPlanner()
        teacher, _ = self._make_teacher(planner, replan_interval=1)
        position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        teacher.label(position, np.array([1.0, 0.0, 0.0]), [position])

        guide = planner.routes[0][1]
        self.assertGreater(float(guide[1]), 0.0)
        self.assertLess(float(guide[1]), 0.2)
        self.assertFalse(np.allclose(guide, [2.0, 0.45, 1.0]))

    def test_planner_failure_masks_waypoints_without_nearest_expert_labels(self):
        planner = _RecordingPlanner(fail=True)
        teacher, fallback = self._make_teacher(planner)
        position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        waypoints, valid, completion = teacher.label(
            position, np.array([1.0, 0.0, 0.0]), [position]
        )

        np.testing.assert_allclose(waypoints, 0.0)
        np.testing.assert_array_equal(valid, np.zeros(5, dtype=np.float32))
        self.assertEqual(completion, 1)
        self.assertEqual(fallback.label_calls, 0)
        self.assertEqual(teacher.last_label_source, "invalid")
        self.assertEqual(teacher.last_label_weight, 0.0)
        stats = teacher.stats()
        self.assertEqual(stats["replan_attempts"], 1)
        self.assertEqual(stats["replan_failures"], 1)
        self.assertEqual(stats["fallback_frames"], 1)
        self.assertEqual(stats["sdf_source"], "precomputed_npz")
        self.assertTrue(stats["optimizer_enabled"])

    def test_failed_replan_keeps_a_safe_cached_path(self):
        planner = _RecordingPlanner()
        teacher, fallback = self._make_teacher(planner, replan_interval=1)
        position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        teacher.label(position, np.array([1.0, 0.0, 0.0]), [position])
        planner.fail = True
        waypoints, valid, completion = teacher.label(
            np.array([0.1, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            [position, np.array([0.1, 0.0, 1.0], dtype=np.float32)],
        )

        self.assertTrue(np.any(valid))
        self.assertEqual(completion, 1)
        self.assertEqual(fallback.label_calls, 0)
        self.assertEqual(teacher.last_label_source, "fast_planner_cache")
        self.assertEqual(teacher.stats()["cached_after_failed_replan_frames"], 1)

    def test_around_waypoints_do_not_skip_past_guide_before_crossing(self):
        planner = _GuideCrossingPlanner()
        teacher, _ = self._make_teacher(planner, replan_interval=1)
        position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        waypoints, valid, _ = teacher.label(
            position, np.array([1.0, 0.0, 0.0]), [position]
        )

        np.testing.assert_allclose(waypoints[:3, 0], [0.9, 1.8, 2.0], atol=1e-6)
        np.testing.assert_array_equal(valid, [1.0, 1.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
