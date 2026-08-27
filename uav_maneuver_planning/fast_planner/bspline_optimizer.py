import numpy as np
from scipy.optimize import minimize


class BsplineOptimizer:
    SMOOTHNESS = 1 << 0
    DISTANCE = 1 << 1
    FEASIBILITY = 1 << 2
    ENDPOINT = 1 << 3
    GUIDE = 1 << 4
    WAYPOINTS = 1 << 6

    GUIDE_PHASE = SMOOTHNESS | GUIDE
    NORMAL_PHASE = SMOOTHNESS | DISTANCE | FEASIBILITY

    def __init__(
        self,
        edt_environment,
        max_vel: float,
        max_acc: float,
        dist0: float = 0.4,
        lambda_smooth: float = 10.0,
        lambda_distance: float = 5.0,
        lambda_feasibility: float = 1e-5,
        lambda_endpoint: float = 0.01,
        lambda_guide: float = 100.0,
        order: int = 3,
        max_iter: int = 80,
    ):
        self.edt_environment = edt_environment
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dist0 = dist0
        self.lambda_smooth = lambda_smooth
        self.lambda_distance = lambda_distance
        self.lambda_feasibility = lambda_feasibility
        self.lambda_endpoint = lambda_endpoint
        self.lambda_guide = lambda_guide
        self.order = order
        self.max_iter = max_iter
        self.guide_pts = None

    def set_guide_path(self, guide_pts):
        self.guide_pts = None if guide_pts is None else np.asarray(guide_pts, dtype=float)

    def bspline_optimize_traj(self, points, ts: float, cost_function: int, max_num_id=1, max_time_id=1):
        del max_num_id, max_time_id
        control_points = np.asarray(points, dtype=float).copy()
        if control_points.shape[0] <= 2 * self.order:
            return control_points
        pt_num, dim = control_points.shape
        start = self.order
        stop = pt_num - self.order
        if cost_function & self.ENDPOINT:
            stop = pt_num
        variable_slice = slice(start, stop)
        if stop <= start:
            return control_points
        initial = control_points[variable_slice].reshape(-1)
        lower = initial - 10.0
        upper = initial + 10.0

        def unpack(x):
            q = control_points.copy()
            q[variable_slice] = np.asarray(x, dtype=float).reshape(-1, dim)
            return q

        def objective(x):
            q = unpack(x)
            cost, gradient = self._cost_and_gradient(q, ts, cost_function)
            return cost, gradient[variable_slice].reshape(-1)

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper)),
            options={"maxiter": self.max_iter, "ftol": 1e-5},
        )
        if result.success or np.isfinite(result.fun):
            control_points[variable_slice] = result.x.reshape(-1, dim)
        return control_points

    def _cost(self, q, ts: float, cost_function: int) -> float:
        cost, _ = self._cost_and_gradient(q, ts, cost_function)
        return cost

    def _cost_and_gradient(self, q, ts: float, cost_function: int):
        cost = 0.0
        gradient = np.zeros_like(q, dtype=float)
        if cost_function & self.SMOOTHNESS:
            if q.shape[0] >= 4:
                jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
                cost += self.lambda_smooth * float(np.sum(jerk * jerk))
                scaled = 2.0 * self.lambda_smooth * jerk
                gradient[:-3] -= scaled
                gradient[1:-2] += 3.0 * scaled
                gradient[2:-1] -= 3.0 * scaled
                gradient[3:] += scaled
        if cost_function & self.DISTANCE:
            end_idx = q.shape[0] - self.order
            for index in range(self.order, end_idx):
                dist, dist_gradient = self.edt_environment.evaluate_edt_with_grad(
                    q[index], -1.0
                )
                if dist < self.dist0:
                    residual = dist - self.dist0
                    cost += self.lambda_distance * residual * residual
                    gradient[index] += (
                        2.0 * self.lambda_distance * residual * np.asarray(dist_gradient)
                    )
        if cost_function & self.FEASIBILITY:
            if q.shape[0] >= 2:
                vel = (q[1:] - q[:-1]) / ts
                excess = np.maximum(np.abs(vel) - self.max_vel, 0.0)
                cost += self.lambda_feasibility * float(np.sum(excess * excess))
                vel_gradient = (
                    2.0 * self.lambda_feasibility * excess * np.sign(vel) / ts
                )
                gradient[:-1] -= vel_gradient
                gradient[1:] += vel_gradient
            if q.shape[0] >= 3:
                ts_squared = ts * ts
                acc = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / ts_squared
                excess = np.maximum(np.abs(acc) - self.max_acc, 0.0)
                cost += self.lambda_feasibility * float(np.sum(excess * excess))
                acc_gradient = (
                    2.0 * self.lambda_feasibility * excess * np.sign(acc) / ts_squared
                )
                gradient[:-2] += acc_gradient
                gradient[1:-1] -= 2.0 * acc_gradient
                gradient[2:] += acc_gradient
        if cost_function & self.GUIDE and self.guide_pts is not None:
            count = min(len(self.guide_pts), max(q.shape[0] - 2 * self.order, 0))
            if count > 0:
                segment = q[self.order : self.order + count]
                residual = segment - self.guide_pts[:count]
                cost += self.lambda_guide * float(np.sum(residual * residual))
                gradient[self.order : self.order + count] += (
                    2.0 * self.lambda_guide * residual
                )
        return float(cost), gradient

    def _smoothness_cost(self, q) -> float:
        if q.shape[0] < 4:
            return 0.0
        jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
        return float(np.sum(jerk * jerk))

    def _distance_cost(self, q) -> float:
        cost = 0.0
        end_idx = q.shape[0] - self.order
        for point in q[self.order : end_idx]:
            dist, _ = self.edt_environment.evaluate_edt_with_grad(point, -1.0)
            if dist < self.dist0:
                cost += (dist - self.dist0) ** 2
        return float(cost)

    def _feasibility_cost(self, q, ts: float) -> float:
        cost = 0.0
        if q.shape[0] >= 2:
            vel = (q[1:] - q[:-1]) / ts
            excess = np.maximum(np.abs(vel) - self.max_vel, 0.0)
            cost += float(np.sum(excess * excess))
        if q.shape[0] >= 3:
            acc = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (ts * ts)
            excess = np.maximum(np.abs(acc) - self.max_acc, 0.0)
            cost += float(np.sum(excess * excess))
        return cost

    def _guide_cost(self, q) -> float:
        if self.guide_pts is None or len(self.guide_pts) == 0:
            return 0.0
        guide = self.guide_pts
        count = min(len(guide), max(q.shape[0] - 2 * self.order, 0))
        if count <= 0:
            return 0.0
        segment = q[self.order : self.order + count]
        return float(np.sum((segment - guide[:count]) ** 2))
