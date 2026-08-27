import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KinodynamicAstarParams:
    max_tau: float = 0.6
    init_max_tau: float = 0.8
    max_vel: float = 3.0
    max_acc: float = 2.0
    w_time: float = 10.0
    horizon: float = 20.0
    lambda_heu: float = 5.0
    resolution_astar: float = 0.2
    time_resolution: float = 0.8
    allocate_num: int = 100000
    max_iter: int = 2000
    check_num: int = 5
    optimistic: bool = True


@dataclass
class PathNode:
    state: np.ndarray
    index: Tuple[int, int, int]
    g_score: float
    f_score: float
    input: np.ndarray
    duration: float
    parent: Optional["PathNode"] = None
    node_state: str = "open"
    time: float = 0.0
    time_idx: int = 0


class KinodynamicAstar:
    REACH_HORIZON = 1
    REACH_END = 2
    NO_PATH = 3
    NEAR_END = 4

    def __init__(self, edt_environment, params: KinodynamicAstarParams):
        self.edt_environment = edt_environment
        self.sdf_map = edt_environment.sdf_map
        self.params = params
        self.tie_breaker = 1.0 + 1.0 / 10000.0
        self.origin, self.map_size = self.sdf_map.get_region()
        self.inv_resolution = 1.0 / params.resolution_astar
        self.inv_time_resolution = 1.0 / params.time_resolution
        self.reset()

    def reset(self) -> None:
        self.expanded_nodes: Dict[Tuple[int, int, int], PathNode] = {}
        self.open_set = []
        self.path_nodes: List[PathNode] = []
        self.use_node_num = 0
        self.iter_num = 0
        self.is_shot_succ = False
        self.coef_shot = None
        self.t_shot = 0.0
        self._tie = 0

    def search(self, start_pt, start_vel, start_acc, end_pt, end_vel, init=True, dynamic=False, time_start=-1.0):
        del dynamic, time_start
        start_pt = np.asarray(start_pt, dtype=float)
        start_vel = np.asarray(start_vel, dtype=float)
        start_acc = np.asarray(start_acc, dtype=float)
        end_pt = np.asarray(end_pt, dtype=float)
        end_vel = np.asarray(end_vel, dtype=float)
        self.start_vel = start_vel
        self.start_acc = start_acc

        if self.sdf_map.get_inflate_occupancy(start_pt) != 0:
            return self.NO_PATH
        if self.sdf_map.get_inflate_occupancy(end_pt) != 0:
            return self.NO_PATH

        start_state = np.r_[start_pt, start_vel]
        end_state = np.r_[end_pt, end_vel]
        time_to_goal = 0.0
        h, _ = self.estimate_heuristic(start_state, end_state)
        start_node = PathNode(
            state=start_state,
            index=self.pos_to_index(start_pt),
            g_score=0.0,
            f_score=self.params.lambda_heu * h,
            input=np.zeros(3),
            duration=0.0,
        )
        self._push_node(start_node)
        end_index = self.pos_to_index(end_pt)
        tolerance = int(math.ceil(1.0 / self.params.resolution_astar))
        init_search = bool(init)

        while self.open_set:
            _, _, cur_node = heapq.heappop(self.open_set)
            if cur_node.node_state == "closed":
                continue

            cur_pos = cur_node.state[:3]
            reach_horizon = np.linalg.norm(cur_pos - start_pt) >= self.params.horizon
            near_end = all(abs(cur_node.index[i] - end_index[i]) <= tolerance for i in range(3))

            if reach_horizon or near_end:
                self.retrieve_path(cur_node)
                if near_end:
                    _, time_to_goal = self.estimate_heuristic(cur_node.state, end_state)
                    self.compute_shot_traj(cur_node.state, end_state, time_to_goal)
                if reach_horizon:
                    return self.REACH_END if self.is_shot_succ else self.REACH_HORIZON
                if near_end:
                    if self.is_shot_succ:
                        return self.REACH_END
                    # A failed direct shot from the initial node does not prove
                    # that no path exists: an obstacle may require a short
                    # detour even though the goal lies within the coarse
                    # near-end voxel tolerance. Expand the start normally and
                    # only accept NEAR_END after making actual progress.
                    if cur_node.parent is not None:
                        return self.NEAR_END

            cur_node.node_state = "closed"
            self.iter_num += 1
            if self.iter_num >= self.params.max_iter:
                return self.NO_PATH
            inputs, durations = self._expansion_inputs(init_search)
            init_search = False
            tmp_expand_nodes: List[PathNode] = []

            for um in inputs:
                for tau in durations:
                    pro_state = self.state_transit(cur_node.state, um, tau)
                    pro_pos = pro_state[:3]
                    if not self.sdf_map.is_in_map(pro_pos):
                        continue
                    pro_id = self.pos_to_index(pro_pos)
                    pro_node = self.expanded_nodes.get(pro_id)
                    if pro_node is not None and pro_node.node_state == "closed":
                        continue
                    pro_vel = pro_state[3:]
                    if np.any(np.abs(pro_vel) > self.params.max_vel):
                        continue
                    if np.linalg.norm(np.asarray(pro_id) - np.asarray(cur_node.index)) == 0:
                        continue
                    if not self._primitive_is_safe(cur_node.state, um, tau):
                        continue

                    tmp_g_score = (float(np.dot(um, um)) + self.params.w_time) * tau + cur_node.g_score
                    heu, _ = self.estimate_heuristic(pro_state, end_state)
                    tmp_f_score = tmp_g_score + self.params.lambda_heu * heu

                    if self._pruned_by_sibling(tmp_expand_nodes, pro_id, tmp_f_score, tmp_g_score, pro_state, um, tau, cur_node):
                        continue

                    if pro_node is None:
                        pro_node = PathNode(
                            state=pro_state,
                            index=pro_id,
                            g_score=tmp_g_score,
                            f_score=tmp_f_score,
                            input=um,
                            duration=tau,
                            parent=cur_node,
                        )
                        self._push_node(pro_node)
                        tmp_expand_nodes.append(pro_node)
                        self.use_node_num += 1
                        if self.use_node_num >= self.params.allocate_num:
                            return self.NO_PATH
                    elif pro_node.node_state == "open" and tmp_g_score < pro_node.g_score:
                        pro_node.state = pro_state
                        pro_node.f_score = tmp_f_score
                        pro_node.g_score = tmp_g_score
                        pro_node.input = um
                        pro_node.duration = tau
                        pro_node.parent = cur_node
                        self._tie += 1
                        heapq.heappush(self.open_set, (pro_node.f_score, self._tie, pro_node))

        return self.NO_PATH

    def get_kino_traj(self, delta_t: float) -> List[np.ndarray]:
        if not self.path_nodes:
            return []
        state_list: List[np.ndarray] = []
        node = self.path_nodes[-1]
        while node.parent is not None:
            duration = node.duration
            t = duration
            while t >= -1e-5:
                state_list.append(self.state_transit(node.parent.state, node.input, t)[:3])
                t -= delta_t
            node = node.parent
        state_list.reverse()
        if self.is_shot_succ and self.coef_shot is not None:
            t = delta_t
            while t <= self.t_shot + 1e-5:
                powers = np.array([1.0, t, t * t, t * t * t])
                state_list.append(self.coef_shot @ powers)
                t += delta_t
        return state_list

    def get_samples(self, ts: float):
        points = self.get_kino_traj(ts)
        if len(points) < 2:
            return points, [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
        start_vel = self.path_nodes[0].state[3:] if self.path_nodes else np.zeros(3)
        end_vel = np.zeros(3)
        start_acc = self.start_acc if hasattr(self, "start_acc") else np.zeros(3)
        end_acc = np.zeros(3)
        return points, [start_vel, end_vel, start_acc, end_acc]

    def pos_to_index(self, pos) -> Tuple[int, int, int]:
        idx = np.floor((np.asarray(pos, dtype=float) - self.origin) * self.inv_resolution).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def time_to_index(self, time: float) -> int:
        return int(math.floor((time) * self.inv_time_resolution))

    def retrieve_path(self, end_node: PathNode) -> None:
        nodes = []
        node = end_node
        while node is not None:
            nodes.append(node)
            node = node.parent
        self.path_nodes = list(reversed(nodes))

    def estimate_heuristic(self, x1, x2):
        dp = x2[:3] - x1[:3]
        v0 = x1[3:]
        v1 = x2[3:]
        c1 = -36.0 * float(np.dot(dp, dp))
        c2 = 24.0 * float(np.dot(v0 + v1, dp))
        c3 = -4.0 * float(np.dot(v0, v0) + np.dot(v0, v1) + np.dot(v1, v1))
        c4 = 0.0
        c5 = self.params.w_time
        candidates = _real_positive_roots([c5, c4, c3, c2, c1])
        v_max = self.params.max_vel * 0.5
        t_bar = float(np.linalg.norm(x1[:3] - x2[:3], ord=np.inf) / max(v_max, 1e-6))
        candidates.append(t_bar)
        best_cost = math.inf
        best_time = t_bar
        for t in candidates:
            if t < t_bar or t <= 1e-6:
                continue
            cost = -c1 / (3.0 * t**3) - c2 / (2.0 * t * t) - c3 / t + self.params.w_time * t
            if cost < best_cost:
                best_cost = cost
                best_time = t
        return (1.0 + self.tie_breaker) * best_cost, best_time

    def compute_shot_traj(self, state1, state2, time_to_goal) -> bool:
        if time_to_goal <= 1e-6:
            return False
        p0 = state1[:3]
        dp = state2[:3] - p0
        v0 = state1[3:]
        v1 = state2[3:]
        dv = v1 - v0
        t_d = time_to_goal
        a = (-12.0 / (t_d**3) * (dp - v0 * t_d) + 6.0 / (t_d * t_d) * dv) / 6.0
        b = 0.5 * (6.0 / (t_d * t_d) * (dp - v0 * t_d) - 2.0 / t_d * dv)
        coef = np.column_stack([p0, v0, b, a])
        t_delta = t_d / 10.0
        t = t_delta
        while t <= t_d + 1e-6:
            powers = np.array([1.0, t, t * t, t * t * t])
            coord = coef @ powers
            vel = coef @ np.array([0.0, 1.0, 2.0 * t, 3.0 * t * t])
            acc = coef @ np.array([0.0, 0.0, 2.0, 6.0 * t])
            if not self.sdf_map.is_in_map(coord):
                return False
            if np.any(np.abs(vel) > self.params.max_vel * 1.5) or np.any(np.abs(acc) > self.params.max_acc * 1.5):
                return False
            if self.sdf_map.get_inflate_occupancy(coord) == 1:
                return False
            t += t_delta
        self.coef_shot = coef
        self.t_shot = t_d
        self.is_shot_succ = True
        return True

    def state_transit(self, state0, um, tau):
        state1 = np.empty(6, dtype=float)
        state1[:3] = state0[:3] + state0[3:] * tau + 0.5 * um * tau * tau
        state1[3:] = state0[3:] + um * tau
        return state1

    def _push_node(self, node: PathNode) -> None:
        self.expanded_nodes[node.index] = node
        self._tie += 1
        heapq.heappush(self.open_set, (node.f_score, self._tie, node))

    def _expansion_inputs(self, init_search: bool):
        if init_search:
            inputs = [self.start_acc.copy()]
            step = self.params.init_max_tau / 20.0
            durations = [step * i for i in range(1, 21)]
            return inputs, durations
        vals = np.arange(-self.params.max_acc, self.params.max_acc + 1e-3, self.params.max_acc / 2.0)
        inputs = [np.array([ax, ay, az], dtype=float) for ax in vals for ay in vals for az in vals]
        durations = [self.params.max_tau * 0.5, self.params.max_tau]
        return inputs, durations

    def _primitive_is_safe(self, cur_state, um, tau) -> bool:
        for k in range(1, self.params.check_num + 1):
            dt = tau * float(k) / float(self.params.check_num)
            pos = self.state_transit(cur_state, um, dt)[:3]
            if self.sdf_map.get_inflate_occupancy(pos) != 0:
                return False
        return True

    def _pruned_by_sibling(self, siblings, pro_id, f_score, g_score, state, um, tau, parent) -> bool:
        for node in siblings:
            if node.index == pro_id:
                if f_score < node.f_score:
                    node.f_score = f_score
                    node.g_score = g_score
                    node.state = state
                    node.input = um
                    node.duration = tau
                    node.parent = parent
                return True
        return False


def _real_positive_roots(coeffs):
    roots = np.roots(coeffs)
    result = []
    for root in roots:
        if abs(root.imag) < 1e-7 and root.real > 1e-6:
            result.append(float(root.real))
    return result
