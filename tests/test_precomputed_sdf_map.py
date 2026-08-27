import unittest

import numpy as np

from uav_maneuver_planning.fast_planner.planner_manager import (
    FastPlannerConfig,
    FastPlannerManager,
)
from uav_maneuver_planning.fast_planner.sdf_map import SDFMap
from uav_maneuver_planning.scenes import create_scene


class PrecomputedSDFMapTests(unittest.TestCase):
    def test_loads_zyx_field_in_world_coordinates_and_applies_inflation(self):
        grid_zyx = np.full((2, 3, 4), 2.0, dtype=np.float32)
        grid_zyx[1, 0, 2] = -0.25
        grid_zyx[0, 1, 1] = 0.20

        sdf_map = SDFMap.from_precomputed(
            grid_zyx=grid_zyx,
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            inflation=0.15,
            bounds_points=((-1.5, -1.5, -1.5), (5.5, 4.5, 3.5)),
            bounds_margin=0.0,
        )

        self.assertAlmostEqual(sdf_map.get_resolution(), 1.0)
        self.assertTrue(sdf_map.is_in_map((-1.49, -1.49, -1.49)))
        self.assertTrue(sdf_map.is_in_map((5.5, 4.5, 3.5)))
        self.assertEqual(sdf_map.get_inflate_occupancy((2.5, 0.5, 1.5)), 1)
        self.assertAlmostEqual(sdf_map.get_distance((2.5, 0.5, 1.5)), -0.40, places=6)
        self.assertEqual(sdf_map.get_inflate_occupancy((1.5, 1.5, 0.5)), 0)
        self.assertAlmostEqual(sdf_map.get_distance((1.5, 1.5, 0.5)), 0.05, places=6)
        self.assertEqual(sdf_map.get_inflate_occupancy((-1.0, -1.0, -1.0)), 0)

    def test_manager_uses_injected_map_without_rebuilding_scene_sdf(self):
        grid_zyx = np.full((4, 4, 4), 2.0, dtype=np.float32)
        sdf_map = SDFMap.from_precomputed(
            grid_zyx=grid_zyx,
            origin=(0.0, 0.0, 0.0),
            voxel_size=0.5,
            inflation=0.15,
            bounds_points=((-1.0, -1.0, -1.0), (3.0, 3.0, 3.0)),
            bounds_margin=0.0,
        )

        manager = FastPlannerManager(
            create_scene(start=(0.0, 0.0, 1.0), goal=(2.0, 0.0, 1.0)),
            FastPlannerConfig(optimize=True),
            sdf_map=sdf_map,
        )

        self.assertIs(manager.sdf_map, sdf_map)
        self.assertIs(manager.edt_environment.sdf_map, sdf_map)

    def test_waypoint_plan_forwards_the_measured_start_state_to_first_segment(self):
        grid_zyx = np.full((4, 4, 4), 2.0, dtype=np.float32)
        sdf_map = SDFMap.from_precomputed(
            grid_zyx=grid_zyx,
            origin=(0.0, 0.0, 0.0),
            voxel_size=0.5,
            inflation=0.15,
            bounds_points=((-1.0, -1.0, -1.0), (3.0, 3.0, 3.0)),
        )
        manager = FastPlannerManager(
            create_scene(start=(0.0, 0.0, 1.0), goal=(2.0, 0.0, 1.0)),
            FastPlannerConfig(optimize=False),
            sdf_map=sdf_map,
        )
        calls = []

        def record_replan(start, start_vel, start_acc, end, end_vel):
            calls.append((start_vel.copy(), start_acc.copy()))
            return [tuple(start), tuple(end)]

        manager.kinodynamic_replan = record_replan
        manager.plan_waypoints(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)],
            start_vel=(0.8, 0.1, 0.0),
            start_acc=(0.2, 0.0, 0.0),
        )

        np.testing.assert_allclose(calls[0][0], [0.8, 0.1, 0.0])
        np.testing.assert_allclose(calls[0][1], [0.2, 0.0, 0.0])
        np.testing.assert_allclose(calls[1][0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(calls[1][1], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
