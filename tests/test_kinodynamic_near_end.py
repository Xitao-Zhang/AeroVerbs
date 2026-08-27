import unittest

import numpy as np

from uav_maneuver_planning.fast_planner.kinodynamic_astar import (
    KinodynamicAstar,
    KinodynamicAstarParams,
)


class _FreeMap:
    def get_region(self):
        return np.asarray([-2.0, -2.0, -2.0]), np.asarray([4.0, 4.0, 4.0])

    def get_inflate_occupancy(self, point):
        return 0

    def is_in_map(self, point):
        return True


class _Environment:
    sdf_map = _FreeMap()


class _ShotAlwaysFailsAstar(KinodynamicAstar):
    def compute_shot_traj(self, state1, state2, time_to_goal):
        return False


class KinodynamicNearEndTests(unittest.TestCase):
    def test_failed_shot_at_start_expands_search_instead_of_returning_no_path(self):
        finder = _ShotAlwaysFailsAstar(
            _Environment(),
            KinodynamicAstarParams(resolution_astar=0.2, max_iter=100),
        )

        status = finder.search(
            np.asarray([0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.asarray([0.8, 0.0, 0.0]),
            np.zeros(3),
            init=False,
        )

        self.assertNotEqual(status, KinodynamicAstar.NO_PATH)
        self.assertGreater(finder.iter_num, 0)


if __name__ == '__main__':
    unittest.main()
