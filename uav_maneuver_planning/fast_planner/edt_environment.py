import numpy as np


class EDTEnvironment:
    """Fast-Planner-style EDTEnvironment wrapper for a static SDFMap."""

    def __init__(self, sdf_map):
        self.sdf_map = sdf_map
        self.resolution_inv = 1.0 / sdf_map.get_resolution()

    def evaluate_edt_with_grad(self, pos, time=-1.0):
        del time
        return self.sdf_map.get_dist_with_grad_trilinear(pos)

    def evaluate_coarse_edt(self, pos, time=-1.0):
        del time
        return self.sdf_map.get_distance(np.asarray(pos, dtype=float))

    def get_map_region(self):
        return self.sdf_map.get_region()
