import math
from typing import Iterable, List, Tuple

import numpy as np


class NonUniformBspline:
    def __init__(self, control_points, order: int = 3, interval: float = 0.2):
        self.set_uniform_bspline(control_points, order, interval)
        self.limit_vel = math.inf
        self.limit_acc = math.inf
        self.limit_ratio = 1.1

    def set_uniform_bspline(self, control_points, order: int, interval: float) -> None:
        self.control_points = np.asarray(control_points, dtype=float)
        self.order = int(order)
        self.interval = float(interval)
        if self.control_points.ndim != 2:
            raise ValueError("control_points must be a 2D array")
        if self.interval <= 0.0:
            raise ValueError("interval must be positive")
        self.n = self.control_points.shape[0] - 1
        self.m = self.n + self.order + 1
        self.knots = np.zeros(self.m + 1, dtype=float)
        for i in range(self.m + 1):
            if i <= self.order:
                self.knots[i] = float(-self.order + i) * self.interval
            else:
                self.knots[i] = self.knots[i - 1] + self.interval

    def evaluate_de_boor(self, u: float) -> np.ndarray:
        u = min(max(self.knots[self.order], float(u)), self.knots[self.m - self.order])
        k = self.order
        while k + 1 < len(self.knots) and self.knots[k + 1] < u:
            k += 1
        d = [self.control_points[k - self.order + i].copy() for i in range(self.order + 1)]
        for r in range(1, self.order + 1):
            for i in range(self.order, r - 1, -1):
                denom = self.knots[i + 1 + k - r] - self.knots[i + k - self.order]
                alpha = 0.0 if abs(denom) < 1e-12 else (u - self.knots[i + k - self.order]) / denom
                d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
        return d[self.order]

    def evaluate_de_boor_t(self, t: float) -> np.ndarray:
        return self.evaluate_de_boor(float(t) + self.knots[self.order])

    def get_time_sum(self) -> float:
        return float(self.knots[self.m - self.order] - self.knots[self.order])

    def get_derivative_control_points(self) -> np.ndarray:
        ctp = np.zeros((self.control_points.shape[0] - 1, self.control_points.shape[1]), dtype=float)
        for i in range(ctp.shape[0]):
            denom = self.knots[i + self.order + 1] - self.knots[i + 1]
            ctp[i] = self.order * (self.control_points[i + 1] - self.control_points[i]) / denom
        return ctp

    def get_derivative(self):
        derivative = NonUniformBspline(self.get_derivative_control_points(), self.order - 1, self.interval)
        derivative.knots = self.knots[1:-1].copy()
        return derivative

    def set_physical_limits(self, vel: float, acc: float) -> None:
        self.limit_vel = float(vel)
        self.limit_acc = float(acc)
        self.limit_ratio = 1.1

    def check_feasibility(self) -> bool:
        return self.check_ratio() <= 1.0 + 1e-4

    def check_ratio(self) -> float:
        max_vel = 0.0
        max_acc = 0.0
        p = self.order
        q = self.control_points
        for i in range(q.shape[0] - 1):
            denom = self.knots[i + p + 1] - self.knots[i + 1]
            vel = p * (q[i + 1] - q[i]) / denom
            max_vel = max(max_vel, float(np.max(np.abs(vel))))
        for i in range(q.shape[0] - 2):
            denom1 = self.knots[i + p + 2] - self.knots[i + 2]
            denom2 = self.knots[i + p + 1] - self.knots[i + 1]
            denom3 = self.knots[i + p + 1] - self.knots[i + 2]
            acc = p * (p - 1) * ((q[i + 2] - q[i + 1]) / denom1 - (q[i + 1] - q[i]) / denom2) / denom3
            max_acc = max(max_acc, float(np.max(np.abs(acc))))
        vel_ratio = max_vel / self.limit_vel if self.limit_vel > 0.0 else 0.0
        acc_ratio = math.sqrt(max_acc / self.limit_acc) if self.limit_acc > 0.0 else 0.0
        return max(vel_ratio, acc_ratio)

    def lengthen_time(self, ratio: float) -> None:
        self.interval *= float(ratio)
        self.set_uniform_bspline(self.control_points, self.order, self.interval)

    def reallocate_time(self) -> bool:
        ratio = min(max(self.check_ratio(), 1.0), self.limit_ratio)
        if ratio <= 1.0 + 1e-4:
            return True
        self.lengthen_time(ratio)
        return self.check_feasibility()

    def sample(self, step: float) -> List[Tuple[float, float, float]]:
        duration = self.get_time_sum()
        count = max(2, int(math.ceil(duration / step)) + 1)
        points = [self.evaluate_de_boor_t(t) for t in np.linspace(0.0, duration, count)]
        return [tuple(float(v) for v in point[:3]) for point in points]

    @staticmethod
    def parameterize_to_bspline(ts: float, point_set: Iterable, start_end_derivative: Iterable) -> np.ndarray:
        point_set = [np.asarray(point, dtype=float) for point in point_set]
        derivatives = [np.asarray(value, dtype=float) for value in start_end_derivative]
        if ts <= 0.0:
            raise ValueError("ts must be positive")
        if len(point_set) < 2:
            raise ValueError("point_set must contain at least two points")
        if len(derivatives) != 4:
            raise ValueError("start_end_derivative must contain v0, v1, a0, a1")
        k = len(point_set)
        a = np.zeros((k + 4, k + 2), dtype=float)
        prow = np.array([1.0, 4.0, 1.0])
        vrow = np.array([-1.0, 0.0, 1.0])
        arow = np.array([1.0, -2.0, 1.0])
        for i in range(k):
            a[i, i : i + 3] = prow / 6.0
        a[k, 0:3] = vrow / (2.0 * ts)
        a[k + 1, k - 1 : k + 2] = vrow / (2.0 * ts)
        a[k + 2, 0:3] = arow / (ts * ts)
        a[k + 3, k - 1 : k + 2] = arow / (ts * ts)
        b = np.vstack([point_set, derivatives])
        ctrl, *_ = np.linalg.lstsq(a, b, rcond=None)
        return ctrl
