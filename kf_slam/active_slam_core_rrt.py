#!/usr/bin/env python3
"""
This code is the core of active slam algorithm for rrt planning. It is a ROS node that administrates the whole process.

"""

import argparse
import csv
import os
import rospy
from std_msgs.msg import Float64 as debug
from std_msgs.msg import Float32 as divergence
from std_msgs.msg import Bool
from geometry_msgs.msg import Point as dot_command
import subprocess
import ast
import time
import sys
import matplotlib.pyplot as plt


import numpy as np
from  rrt_wrapper import RRT_star_ROS
from GET_data import GET_data
from copy import deepcopy as copy
from simulated_2d_slam2_test import Simulator_manager
from nav_msgs.msg  import Path
from uncertainty_frointier import UFrontier
from msf_rrt import MSF_RRT

from scipy.ndimage import distance_transform_edt # implementa la SDF sobre un mapa de ocupacion.

from visualization_msgs.msg import MarkerArray
from frontiers_markers import create_border_marker,clear_all_markers

path_ready_flag=False
current_divergence=0.0

class Main_data:
    def __init__(self):
        self.P=0
        self.states=0
        self.exp_map=0
        self.obstacle_map=0

def get_states()->Main_data:
    """ Se subscribe a los topics correspondientes y devuelve un objeto del tipo "Main_data" con los datos 
    Covariance, states, exp_map y obstacle map"""
    gt_data=GET_data()
    while not(gt_data.got_states and gt_data.obstacle_map.flag_ and gt_data.exp_map.flag_):
        pass # wait to get the obstacle map
    
    md=Main_data()
    md.P=gt_data.P
    md.states=gt_data.states
    md.exp_map=gt_data.exp_map.map_
    md.obstacle_map=gt_data.obstacle_map.map_.T

    pose_agent=gt_data.states[0:2]
    obstacle_map=gt_data.obstacle_map.map_.T
    del gt_data
    return md# pose_agent[0]=x,pose_agent[1]=y

def compute_path_length(path: Path) -> float:
    if path is None:
        return 0.0
    distance=0.0
    if len(path.poses)==0:
        return distance
    prev=np.array([path.poses[0].pose.position.x,path.poses[0].pose.position.y])
    for pose in path.poses[1:]:
        curr=np.array([pose.pose.position.x,pose.pose.position.y])
        distance+=np.linalg.norm(curr-prev)
        prev=curr
    return distance

def compute_coverage(exp_map: np.ndarray) -> float:
    if exp_map is None or exp_map.size==0:
        return 0.0
    beta=rospy.get_param('sigma_max',1.0)
    return float(np.mean(exp_map<beta))

def compute_uncertainty(exp_map: np.ndarray) -> float:
    if exp_map is None or exp_map.size==0:
        return 0.0
    return float(np.mean(exp_map))

def parse_cli_args():
    parser=argparse.ArgumentParser(description='Active SLAM core with selectable planner')
    parser.add_argument('--planner_type',choices=['baseline','msf_rrt'],default='baseline')
    parser.add_argument('--result_path',default=None,help='CSV file to store run metrics')
    args,_=parser.parse_known_args(rospy.myargv()[1:])
    return args

def init_result_writer(path):
    if not path:
        return None,None
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)
    handler=open(path,'w',newline='')
    writer=csv.writer(handler)
    writer.writerow(['step','coverage','distance','uncertainty','siren'])
    return handler,writer

def get_frontiers(self,u_map=np.array([]),obstacle_map=np.array([]),show_animation=False):
    # Get frontiers from the map and return senters and a flag that indicates if there are frontiers or not
    #import pdb;pdb.set_trace()
    uf=UFrontier(beta=rospy.get_param('beta'),sig_mx=rospy.get_param('sigma_max'),cs=rospy.get_param('cell_size')
                 ,uf_treshold=rospy.get_param('uf_treshold'),show_animation=show_animation)
    
    uf.set_maps(u_map=u_map,obstacle_map=obstacle_map)
    pose_=np.array([self.states[0],self.states[1]])
    # p_objetive,distance,flag=uf.find_frontiers(pose=pose_)
    centers,_,flag=uf.find_frontiers(pose=pose_)
    #D=0
    #if flag:
    #    D=uf.get_aprior_divergence(p_objetive)
    
    return centers,flag    

def prepare_obstacle_map(md,exclusion=0)->np.array:
    """ segmenta el mapa de obstaculos y genera zonas de exclusion para el path planning.

    Args:
        md (_type_): _description_

    Returns:
        np.array: _description_
    """
    min_dist_obstacle=rospy.get_param('min_obstacle_distance_sdf')#1.5 # in meters
    cs=rospy.get_param('cell_size')
    sdf_map = distance_transform_edt(~(md.obstacle_map>0.5)) # 0.6 es el umbral de ocupacion
    modified_obstacle_map=(sdf_map>(min_dist_obstacle+exclusion)/cs) 
    return np.logical_not(modified_obstacle_map).astype(float)*0.7+0.1 # pone 0.8 en los obstaculos y 0.1 en el resto

def point_on_line_distance_from_P2(P1, P2, d_min):
    # chatpgt  realizo este codigo
    # Calcula el vector de dirección de la recta.
    direction_vector = -P2 + P1
    # Normaliza el vector de dirección.
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # Calcula el desplazamiento desde P2 a lo largo de la recta.
    displacement_vector = unit_direction_vector * d_min
    # Calcula el punto deseado en la recta.
    desired_point = P2 + displacement_vector
    return [desired_point[0],desired_point[1]]

def landmarks_objetive(md):
    """Calcula el punto objetivo para cada landmark. 
    
    Si no hay linea de vision directa, puede que el punto generado no este sobre el path... habria que hacerlo con el RRT y definir la 
    distancia final como fov-2.
    """
    fov=5.0
    robot_pose=np.array(md.states[0:2])
    landmarks=md.states[2:]
    #import pdb; pdb.set_trace()
    N=len(landmarks)/2
    points=[]
    for i in range(int(N)):
        l=np.array(landmarks[2*i:2*i+2])
        points.append(point_on_line_distance_from_P2(robot_pose,l,fov-2.0))
        # 
    return points

def path_callback(msg=Bool()):
    global path_ready_flag
    path_ready_flag=msg.data
    time.sleep(30) # para evitar atascos del robot, el sistema lo puede mover automaticamente.
    return
def divergence_callback(msg=divergence()):
    global current_divergence
    current_divergence=msg.data
    return

def divergence_prediction():
    global current_divergence


def process_(planner_type="baseline", msf_params=None, show_animation=False)->(Path,float,Main_data):
    global current_divergence,marker_array
    minimun_distance=rospy.get_param('min_obstacle_distance_sdf')
    max_iter_=rospy.get_param('max_iter_rrt')
    if planner_type=="msf_rrt":
        params=msf_params or {}
        rrt=MSF_RRT(show_animation=False,minimun_distance=minimun_distance,max_iter=max_iter_,**params)
    else:
        rrt=RRT_star_ROS(show_animation=False,minimun_distance=minimun_distance,max_iter=max_iter_)
    md=get_states()
    rrt.robot_pose.x=md.states[0:2][0]
    rrt.robot_pose.y=md.states[0:2][1]
    P=md.P ;    D=np.array([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
    sigma_robot=np.power(np.linalg.det(D),0.25)
    rrt.robot_pose.z=sigma_robot
    rrt.obstacle_map=md.obstacle_map
    rrt.landmarks.landmarks_poses=md.states[2:]
    rrt.landmarks.P=md.P
    rrt.process_landmark()

    goals=[]
    modified_obstacle_map=prepare_obstacle_map(md)
    centers,flag=get_frontiers(md,obstacle_map=modified_obstacle_map.T,u_map=md.exp_map,show_animation=show_animation)
    if not flag:
        print('A goal was not found.')
        return None,None,md

    for c in centers:
        goals.append([c[0],c[1]])

    print('Goals: ',((np.array(goals)).T).round(3))


    first_time=True
    divergences=[]
    divergences_=[]
    paths=[]
    dmin=None
    for g in goals:
        rrt.goal_pose.x=g[0]
        rrt.goal_pose.y=g[1]
        if first_time:
            first_time=False
            rrt.run_first_time()
        else:
            rrt.run_hot()
        if rrt.path is not None:
            path=[]
            d=0
            pose_anterior=np.array([rrt.robot_pose.x,rrt.robot_pose.y])
            for p in rrt.path.poses:
                path.append((p.pose.position.x,p.pose.position.y))
                d=d+np.linalg.norm(np.array([p.pose.position.x,p.pose.position.y])-pose_anterior)
                pose_anterior=np.array([p.pose.position.x,p.pose.position.y])
            dmax=20.0
            dmax_=100

            if dmin is None or d<dmin:
                dmin=d
                if len(divergences)==0:
                    divergences.append(1)
                else:
                    divergences.append(max(divergences)+1)

            else:
                divergences.append(0)
            paths.append(rrt.path)
            divergences_.append(0)
            continue

            if d>dmax+dmax_:
                rospy.logwarn('Distancia de trayectoria mayor a '+str(dmax)+' m')
                divergences.append(0)
                paths.append([])
                divergences_.append(0)
                continue

            time.sleep(1)
            sim=Simulator_manager(ros_required=False)
            sim.states=md.states
            sim.P=md.P
            sim.exp_map=md.exp_map
            sim.obstacle_map=md.obstacle_map.T
            sim.path=path
            diver,distance=sim.run()
            print('Point: ',f'({g[0]:.3f},{g[1]:.3f})','Divergence: ',f'{diver:.3f}',' Distance: ',f'{distance:.3f}')

            del sim
            if distance<2.0:
                divergences.append(0)
                paths.append([])
            else:

                paths.append(rrt.path)

                if distance<dmax+dmax_:
                    divergences.append((diver-current_divergence)*(np.power(np.math.e,-distance/dmax)))
                else:
                    rospy.logwarn('Distancia de trayectoria mayor a '+str(dmax)+' m')
            divergences_.append(diver)
        else:
            print('A path was not found.')
            divergences.append(0)
            divergences_.append(0)
            paths.append([])

    max_value = max(divergences)
    print('rewards: ',np.array(divergences).round(3))

    if max_value==0:
        print('RRT did not find any viable path, recalculating.')
        return -1,None,md
    max_index = divergences.index(max_value)

    clear_all_markers(marker_array,frame_id="robot1_tf/odom_groundtruth")
    marker_data=create_border_marker(centers,frame_id="robot1_tf/odom_groundtruth",id=max_index,radius=0.5)
    marker_array.publish(marker_data)

    return paths[max_index],divergences_[max_index],md

def move_robot_safe():
    global current_divergence
    minimun_distance=rospy.get_param('min_obstacle_distance_sdf')
    max_iter_=rospy.get_param('max_iter_rrt')
    rrt=RRT_star_ROS(show_animation=False,minimun_distance=minimun_distance,max_iter=max_iter_)
    # obtiene la posicion actual del robot
    md=get_states()
    #import pdb; pdb.set_trace()
    rrt.robot_pose.x=md.states[0:2][0]
    rrt.robot_pose.y=md.states[0:2][1]
    P=md.P ;    D=np.array([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
    sigma_robot=np.power(np.linalg.det(D),0.25)
    rrt.robot_pose.z=sigma_robot # temperatura del robot
    rrt.obstacle_map=md.obstacle_map
    # procesa los landmarks para la planificacion.
    rrt.landmarks.landmarks_poses=md.states[2:]
    rrt.landmarks.P=md.P
    rrt.process_landmark()
    goals=[]
    #goals.extend(landmarks_objetive(md)) # TODO, se planifica sin tener en cuenta los landmarks
    modified_obstacle_map=prepare_obstacle_map(md,exclusion=2.0)
    r_pose=(rrt.robot_pose.x/0.1+300,rrt.robot_pose.y/0.1+300) # robot pose en celdas

    # obtener un conjunto de goals cercanos que no tengan obstaculos cerca
    rrt.minimun_distance=0.0
    rrt.goal_pose.x=g[0]
    rrt.goal_pose.y=g[1]

    pass
    
def principal():
    global path_ready_flag, current_divergence,marker_array
    args=parse_cli_args()
    msf_params=rospy.get_param('msf_rrt',{}) if args.planner_type=='msf_rrt' else {}
    result_handle,result_writer=init_result_writer(args.result_path)
    cumulative_distance=0.0
    step_counter=0
    rospy.init_node('active_slam',anonymous=True)
    pub_rrt_path=rospy.Publisher('/path_rrt_star',Path,queue_size=1)

    path_ready=rospy.Subscriber('/path_ready',Bool,callback=path_callback,queue_size=1)
    divergence_reading=rospy.Subscriber('/divergence',divergence,callback=divergence_callback,queue_size=1)
    marker_array = rospy.Publisher('border_markers', MarkerArray, queue_size=10)
    rate=rospy.Rate(1)
    predicted_diver=None
    recompute_flag=0

    try:
        while not rospy.is_shutdown():
            if path_ready_flag or recompute_flag>0:
                if predicted_diver is not None:
                    print('##PREDICTED DIVERGENCE VS CURRENT DIVERGENCE: ',round(current_divergence-predicted_diver,3),'Current: ',round(current_divergence,3),'Predicted: ',round(predicted_diver,3))
                path,predicted_diver,md=process_(planner_type=args.planner_type,msf_params=msf_params)
                if path is None:
                    rospy.signal_shutdown('Task successfully completed.')
                    return
                elif path==-1:
                   rospy.logwarn('A valid path is not found. It is necessary to move the vehicle to find a valid path.')
                   if recompute_flag>10:
                       rospy.logerr('A valid path was searched for 10 times and not found. It is necessary to move the vehicle to find a valid path. The simulation will close: ERROR path not found.')
                       return
                   recompute_flag+=1
                else:
                    pub_rrt_path.publish(path)
                    coverage=compute_coverage(md.exp_map)
                    step_distance=compute_path_length(path)
                    cumulative_distance+=step_distance
                    uncertainty=compute_uncertainty(md.exp_map)
                    siren_score=rospy.get_param('siren_score',None)
                    if result_writer:
                        result_writer.writerow([step_counter,coverage,cumulative_distance,uncertainty,'' if siren_score is None else siren_score])
                        result_handle.flush()
                    step_counter+=1
                    recompute_flag=0

                path_ready_flag=False
            rate.sleep()
    finally:
        if result_handle:
            result_handle.close()

if __name__=='__main__':
    principal()
