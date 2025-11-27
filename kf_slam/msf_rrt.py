#!/usr/bin/env python3
"""
MSF-RRT planner that extends the baseline RRT* ROS wrapper with additional strategies:
- Target bias sampling
- Bias extension
- Adaptive step size based on obstacle clearance
- Optional B-spline smoothing
"""

import math
import random
from copy import deepcopy as copy

import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt

from rrt_planning import RRTStar
from rrt_wrapper import RRT_star_ROS


class AdaptiveRRTStar(RRTStar):
    """RRT* variant with biased sampling and adaptive step sizing."""

    def __init__(
        self,
        *args,
        bias_ratio: float = 0.3,
        step_size_min: float = 0.3,
        step_size_max: float = 0.8,
        obstacle_threshold: float = 0.5,
        cell_size: float = 0.1,
        **kwargs,
    ):
        self.bias_ratio = bias_ratio
        self.step_size_min = step_size_min / cell_size
        self.step_size_max = step_size_max / cell_size
        self.obstacle_threshold = obstacle_threshold / cell_size
        self.cell_size = cell_size
        self.sdf_map = None
        self._last_bias = False
        super().__init__(*args, **kwargs)
        # Override base step size with the maximum allowed; steer will adapt per edge.
        self.step_size = self.step_size_max

    def set_sdf_map(self, sdf_map: np.ndarray):
        self.sdf_map = sdf_map

    def generate_random_node(self):
        if random.random() < self.bias_ratio:
            self._last_bias = True
            return copy(self.goal)
        self._last_bias = False
        return super().generate_random_node()

    def _get_clearance(self, node):
        if self.sdf_map is None:
            return self.obstacle_threshold
        x = int(np.clip(node.x, 0, self.sdf_map.shape[1] - 1))
        y = int(np.clip(node.y, 0, self.sdf_map.shape[0] - 1))
        return self.sdf_map[y, x]

    def _compute_step_size(self, from_node, to_node):
        clearance = min(self._get_clearance(from_node), self._get_clearance(to_node))
        if clearance <= 0:
            return self.step_size_min
        ratio = min(clearance / self.obstacle_threshold, 1.0)
        return self.step_size_min + (self.step_size_max - self.step_size_min) * ratio

    def steer(self, from_node, to_node):
        step_size = self._compute_step_size(from_node, to_node)
        distance = math.sqrt((to_node.x - from_node.x) ** 2 + (to_node.y - from_node.y) ** 2)
        if distance < step_size:
            new_node = copy(to_node)
        else:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_x = from_node.x + step_size * math.cos(theta)
            new_y = from_node.y + step_size * math.sin(theta)
            new_node = type(from_node)(new_x, new_y)

        if self._last_bias and distance > 0 and self.is_collision_free(from_node, new_node):
            # Bias extension: take an extra step towards the goal when biased sampling succeeds.
            theta = math.atan2(to_node.y - new_node.y, to_node.x - new_node.x)
            new_x = new_node.x + min(step_size, self.step_size_max) * math.cos(theta)
            new_y = new_node.y + min(step_size, self.step_size_max) * math.sin(theta)
            candidate = type(from_node)(new_x, new_y)
            if self.is_collision_free(new_node, candidate):
                new_node = candidate
        return new_node


class MSF_RRT(RRT_star_ROS):
    """Modified planner implementing MSF-RRT strategies on top of the baseline wrapper."""

    def __init__(
        self,
        minimun_distance: float = 1.0,
        show_animation: bool = False,
        max_iter: int = 100,
        bias_ratio: float = 0.3,
        step_size_min: float = 0.3,
        step_size_max: float = 0.8,
        obstacle_threshold: float = 0.5,
        smoothing: bool = True,
    ):
        super().__init__(minimun_distance=minimun_distance, show_animation=show_animation, max_iter=max_iter)
        self.bias_ratio = bias_ratio
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        self.obstacle_threshold = obstacle_threshold
        self.smoothing = smoothing

    def _build_rrt(self):
        LOS_map = self.rc.build_LOS_map(self.obstacle_map > 0.4, self.landmarks_list)
        transit_map = self.process_obstacle_map()
        self.shape_ = [transit_map.shape[0] / 2, transit_map.shape[1] / 2]

        sigma_robot = self.robot_pose.z / (self.cell_size * self.Q)
        r_pose = (
            self.robot_pose.x / self.cell_size + self.shape_[0],
            self.robot_pose.y / self.cell_size + self.shape_[1],
        )
        g_pose = (
            self.goal_pose.x / self.cell_size + self.shape_[0],
            self.goal_pose.y / self.cell_size + self.shape_[1],
        )

        sdf_map = distance_transform_edt(~(self.obstacle_map > 0.5))

        self.rrt_module = AdaptiveRRTStar(
            start=r_pose,
            goal=g_pose,
            obstacle_map=transit_map,
            step_size=self.step_size_max / self.cell_size,
            max_iter=self.max_iter,
            nearest=self.nearest_distance / self.cell_size,
            animation=self.show_animation,
            landmarks=self.landmarks_list,
            temperature=sigma_robot,
            LOS_map=LOS_map,
            bias_ratio=self.bias_ratio,
            step_size_min=self.step_size_min,
            step_size_max=self.step_size_max,
            obstacle_threshold=self.obstacle_threshold,
            cell_size=self.cell_size,
        )
        self.rrt_module.set_sdf_map(sdf_map)

    def run_first_time(self):
        if self.show_animation:
            plt.figure(1)
            plt.imshow(self.obstacle_map)
            plt.show()
        self._build_rrt()
        path_ = self.rrt_module.plan()
        self.path_manager(path_)

    def run_hot(self):
        self.run_hot_flag = False
        g_pose = (
            self.goal_pose.x / self.cell_size + self.shape_[0],
            self.goal_pose.y / self.cell_size + self.shape_[1],
        )
        state = self.rrt_module.is_connected(g_pose)
        if state is None:
            self.valid_path = False
            self.compute = False
        else:
            self.path_manager(state)

    def _smooth_path(self, path_points):
        if not self.smoothing or len(path_points) < 4:
            return path_points
        try:
            data = np.array(path_points)
            tck, u = splprep([data[:, 0], data[:, 1]], s=0)
            u_fine = np.linspace(0, 1, num=max(len(path_points) * 3, 12))
            x_new, y_new = splev(u_fine, tck)
            return list(zip(x_new, y_new))
        except Exception:
            return path_points

    def path_manager(self, path_):
        if path_:
            smoothed = self._smooth_path(path_)
            path = np.array(smoothed)
            if self.show_animation:
                print(path)
                plt.plot(path[:, 0], path[:, 1], 'r--')
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.show()

            self.path = Path()
            for l in smoothed:
                p = PoseStamped()
                p.pose.position.x = (l[0] - self.shape_[0]) * self.cell_size
                p.pose.position.y = (l[1] - self.shape_[1]) * self.cell_size
                self.path.poses.append(p)

            self.valid_path = True
            self.compute = False
        else:
            if self.show_animation:
                plt.show()
            self.valid_path = False
            self.compute = False
            self.path = None
