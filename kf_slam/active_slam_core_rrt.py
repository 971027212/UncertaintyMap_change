#!/usr/bin/env python3
"""
Active SLAM core with selectable planner (baseline / msf_rrt)
Includes improved coverage metric: “explored cells = cells whose value changed
compared to the initial exp_map”, matching RViz ‘blue-area’ visualization.
"""

import argparse
import csv
import os
import rospy
from std_msgs.msg import Float64 as debug
from std_msgs.msg import Float32 as divergence
from std_msgs.msg import Bool
from geometry_msgs.msg import Point as dot_command
import time
import sys

import numpy as np
from rrt_wrapper import RRT_star_ROS
from GET_data import GET_data
from simulated_2d_slam2_test import Simulator_manager
from nav_msgs.msg import Path
from uncertainty_frointier import UFrontier
from msf_rrt import MSF_RRT

from scipy.ndimage import distance_transform_edt
from visualization_msgs.msg import MarkerArray
from frontiers_markers import create_border_marker, clear_all_markers

# --------------------------------------------------------------------------
# Global variables
# --------------------------------------------------------------------------
path_ready_flag = False
current_divergence = 0.0

initial_exp_map = None          # exp_map at step 0
explored_mask = None            # boolean mask marking explored cells


# --------------------------------------------------------------------------
# Data holder
# --------------------------------------------------------------------------
class Main_data:
    def __init__(self):
        self.P = 0
        self.states = 0
        self.exp_map = 0
        self.obstacle_map = 0


# --------------------------------------------------------------------------
# State collection
# --------------------------------------------------------------------------
def get_states() -> Main_data:
    gt_data = GET_data()
    while not (gt_data.got_states and gt_data.obstacle_map.flag_ and gt_data.exp_map.flag_):
        pass

    md = Main_data()
    md.P = gt_data.P
    md.states = gt_data.states
    md.exp_map = gt_data.exp_map.map_
    md.obstacle_map = gt_data.obstacle_map.map_.T

    del gt_data
    return md


# --------------------------------------------------------------------------
# Path length
# --------------------------------------------------------------------------
def compute_path_length(path: Path) -> float:
    if path is None:
        return 0.0

    if len(path.poses) == 0:
        return 0.0

    distance = 0.0
    prev = np.array([path.poses[0].pose.position.x,
                     path.poses[0].pose.position.y])

    for pose in path.poses[1:]:
        curr = np.array([pose.pose.position.x, pose.pose.position.y])
        distance += np.linalg.norm(curr - prev)
        prev = curr

    return distance


# --------------------------------------------------------------------------
# *** Improved Coverage Computation ***
# --------------------------------------------------------------------------
def compute_coverage(exp_map: np.ndarray) -> float:

    global initial_exp_map, explored_mask

    if exp_map is None or exp_map.size == 0:
        return 0.0

   
    if initial_exp_map is None:
        initial_exp_map = exp_map.copy()
        explored_mask = np.zeros_like(exp_map, dtype=bool)
        return 0.0

    
    if initial_exp_map.shape != exp_map.shape:
        initial_exp_map = exp_map.copy()
        explored_mask = np.zeros_like(exp_map, dtype=bool)
        return 0.0

    
    delta = np.abs(exp_map - initial_exp_map)

   
    max_delta = float(np.max(delta))
    if max_delta <= 0.0:
        return 0.0

    
    eps = max(1e-6, 0.01 * max_delta)

    newly_explored = delta > eps

    
    explored_mask |= newly_explored

    coverage = float(np.mean(explored_mask))
    return float(np.clip(coverage, 0.0, 1.0))



# --------------------------------------------------------------------------
# Uncertainty tracking
# --------------------------------------------------------------------------
def compute_uncertainty(exp_map: np.ndarray) -> float:
    if exp_map is None or exp_map.size == 0:
        return 0.0
    return float(np.mean(exp_map))


# --------------------------------------------------------------------------
# CLI arguments
# --------------------------------------------------------------------------
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner_type', choices=['baseline', 'msf_rrt'], default='baseline')
    parser.add_argument('--result_path', default=None)
    args, _ = parser.parse_known_args(rospy.myargv()[1:])
    return args


# --------------------------------------------------------------------------
# CSV writer
# --------------------------------------------------------------------------
def init_result_writer(path):
    if not path:
        return None, None
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['step', 'coverage', 'distance', 'uncertainty', 'siren'])
    return f, w


# --------------------------------------------------------------------------
# Frontier detection
# --------------------------------------------------------------------------
def get_frontiers(self, u_map=np.array([]), obstacle_map=np.array([]), show_animation=False):
    uf = UFrontier(
        beta=rospy.get_param('beta'),
        sig_mx=rospy.get_param('sigma_max'),
        cs=rospy.get_param('cell_size'),
        uf_treshold=rospy.get_param('uf_treshold'),
        show_animation=show_animation
    )

    uf.set_maps(u_map=u_map, obstacle_map=obstacle_map)
    pose_ = np.array([self.states[0], self.states[1]])

    centers, _, flag = uf.find_frontiers(pose=pose_)
    return centers, flag


# --------------------------------------------------------------------------
# Obstacle map
# --------------------------------------------------------------------------
def prepare_obstacle_map(md, exclusion=0):
    min_dist = rospy.get_param('min_obstacle_distance_sdf')
    cs = rospy.get_param('cell_size')
    sdf_map = distance_transform_edt(~(md.obstacle_map > 0.5))
    mod = (sdf_map > (min_dist + exclusion) / cs)
    return np.logical_not(mod).astype(float) * 0.7 + 0.1


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def path_callback(msg=Bool()):
    global path_ready_flag
    path_ready_flag = msg.data
    time.sleep(30)
    return


def divergence_callback(msg=divergence()):
    global current_divergence
    current_divergence = msg.data
    return


# --------------------------------------------------------------------------
# Main planning process
# --------------------------------------------------------------------------
def process_(planner_type="baseline", msf_params=None, show_animation=False):
    global current_divergence, marker_array

    min_dist = rospy.get_param('min_obstacle_distance_sdf')
    max_iter_ = rospy.get_param('max_iter_rrt')

    if planner_type == "msf_rrt":
        params = msf_params or {}
        rrt = MSF_RRT(show_animation=False, minimun_distance=min_dist, max_iter=max_iter_, **params)
    else:
        rrt = RRT_star_ROS(show_animation=False, minimun_distance=min_dist, max_iter=max_iter_)

    md = get_states()

    rrt.robot_pose.x = md.states[0]
    rrt.robot_pose.y = md.states[1]

    P = md.P
    D = np.array([[P[0][0], P[0][1]], [P[1][0], P[1][1]]])
    sigma_robot = np.power(np.linalg.det(D), 0.25)
    rrt.robot_pose.z = sigma_robot

    rrt.obstacle_map = md.obstacle_map
    rrt.landmarks.landmarks_poses = md.states[2:]
    rrt.landmarks.P = md.P
    rrt.process_landmark()

    modified_obstacle_map = prepare_obstacle_map(md)
    centers, flag = get_frontiers(md, obstacle_map=modified_obstacle_map.T, u_map=md.exp_map)

    if not flag:
        print("A goal was not found.")
        return None, None, md

    goals = [[c[0], c[1]] for c in centers]
    first_time = True
    divergences = []
    divergences_ = []
    paths = []
    dmin = None

    for g in goals:
        rrt.goal_pose.x = g[0]
        rrt.goal_pose.y = g[1]

        if first_time:
            first_time = False
            rrt.run_first_time()
        else:
            rrt.run_hot()

        if rrt.path is not None:
            paths.append(rrt.path)
            divergences.append(1 if dmin is None else 0)
            divergences_.append(0)
            continue

        divergences.append(0)
        divergences_.append(0)
        paths.append([])

    max_value = max(divergences)
    if max_value == 0:
        print("RRT did not find any viable path, recalculating.")
        return -1, None, md

    max_index = divergences.index(max_value)

    clear_all_markers(marker_array, frame_id="robot1_tf/odom_groundtruth")
    marker_data = create_border_marker(centers, frame_id="robot1_tf/odom_groundtruth", id=max_index, radius=0.5)
    marker_array.publish(marker_data)

    return paths[max_index], divergences_[max_index], md


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def principal():
    global path_ready_flag, current_divergence, marker_array

    args = parse_cli_args()
    rospy.init_node('active_slam', anonymous=True)

    msf_params = rospy.get_param('msf_rrt', {}) if args.planner_type == 'msf_rrt' else {}
    result_handle, result_writer = init_result_writer(args.result_path)

    cumulative_distance = 0.0
    step_counter = 0

    rospy.Subscriber('/path_ready', Bool, path_callback, queue_size=1)
    rospy.Subscriber('/divergence', divergence, divergence_callback, queue_size=1)
    marker_array = rospy.Publisher('border_markers', MarkerArray, queue_size=10)
    pub_rrt_path = rospy.Publisher('/path_rrt_star', Path, queue_size=1)

    rate = rospy.Rate(1)
    predicted_diver = None
    recompute_flag = 0

    try:
        while not rospy.is_shutdown():

            if path_ready_flag or recompute_flag > 0:

                path, predicted_diver, md = process_(
                    planner_type=args.planner_type,
                    msf_params=msf_params
                )

                if path is None:
                    rospy.signal_shutdown("Task completed.")
                    return
                elif path == -1:
                    recompute_flag += 1
                    if recompute_flag > 10:
                        rospy.logerr("Failed for 10 consecutive attempts. Exiting.")
                        return
                else:
                    pub_rrt_path.publish(path)

                    # Distance
                    step_dist = compute_path_length(path)
                    cumulative_distance += step_dist

                    # *** Improved coverage ***
                    coverage = compute_coverage(md.exp_map)

                    # Uncertainty
                    uncertainty = compute_uncertainty(md.exp_map)

                    siren_score = rospy.get_param('siren_score', None)

                    if result_writer:
                        result_writer.writerow([
                            step_counter, coverage, cumulative_distance,
                            uncertainty,
                            '' if siren_score is None else siren_score
                        ])
                        result_handle.flush()

                    step_counter += 1
                    recompute_flag = 0

                path_ready_flag = False

            rate.sleep()

    finally:
        if result_handle:
            result_handle.close()


if __name__ == '__main__':
    principal()

