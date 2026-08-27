import unittest

import numpy as np

from uav_maneuver_planning.fast_planner.bspline_optimizer import BsplineOptimizer


class _LinearDistanceEnvironment:
    def evaluate_edt_with_grad(self, point, time):
        del time
        return float(point[0]), np.array([1.0, 0.0, 0.0], dtype=float)


class BsplineOptimizerGradientTests(unittest.TestCase):
    def test_analytic_gradient_matches_central_difference(self):
        optimizer = BsplineOptimizer(
            _LinearDistanceEnvironment(),
            max_vel=0.6,
            max_acc=1.2,
            dist0=0.4,
            order=1,
        )
        q = np.array(
            [
                [0.60, 0.00, 0.10],
                [0.35, 0.20, 0.05],
                [0.10, -0.15, 0.30],
                [0.25, 0.35, -0.10],
                [0.70, -0.20, 0.15],
                [0.45, 0.10, 0.40],
                [0.80, -0.10, 0.20],
            ],
            dtype=float,
        )
        flags = (
            BsplineOptimizer.SMOOTHNESS
            | BsplineOptimizer.DISTANCE
            | BsplineOptimizer.FEASIBILITY
        )

        cost, analytic = optimizer._cost_and_gradient(q, ts=0.5, cost_function=flags)
        numeric = np.zeros_like(q)
        epsilon = 1e-6
        for index in np.ndindex(q.shape):
            plus = q.copy()
            minus = q.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[index] = (
                optimizer._cost(plus, 0.5, flags) - optimizer._cost(minus, 0.5, flags)
            ) / (2.0 * epsilon)

        self.assertTrue(np.isfinite(cost))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
