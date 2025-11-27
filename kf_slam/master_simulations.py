import subprocess
import time
# import psutil
import sys
import ruamel.yaml
import datetime
import argparse
import os


def run_command(command, output=True):
    """
    Launch a shell command as a subprocess.

    :param command: Shell command string
    :param output: If False, redirect stdout/stderr to /dev/null
    :return: Popen process handle
    """
    if output:
        process = subprocess.Popen(command, shell=True)
    else:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    return process


def update_yaml_file(file_path, new_x, new_y, planner,
                     mpc=False, UF=False, sigma_max=0.6, map='nada'):
    yaml = ruamel.yaml.YAML()

    # Leer el contenido del archivo YAML
    with open(file_path, 'r') as file:
        data = yaml.load(file)

    # Modificar los valores de initial_pose.x e initial_pose.y
    data['initial_pose']['x'] = new_x
    data['initial_pose']['y'] = new_y
    data['planner'] = planner
    data['planner_mpc'] = mpc
    data['UF'] = UF
    data['sigma_max'] = sigma_max
    data['map'] = map

    # Escribir los cambios en el archivo YAML preservando la estructura original
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def read_yaml_file(file_path):
    yaml = ruamel.yaml.YAML()
    with open(file_path, 'r') as file:
        data = yaml.load(file)
    return data


def parse_args():
    """
    Extra CLI options to select planner and result directory.

    --planner_type: 'baseline' or 'msf_rrt'
    --result_dir: folder to store CSV metrics (one per simulation)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--planner_type',
        choices=['baseline', 'msf_rrt'],
        default='baseline',
        help='Which planner to use in active_slam_core_rrt.py'
    )
    parser.add_argument(
        '--result_dir',
        default='./results',
        help='Directory to store CSV metrics (one per simulation).'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # leer la primera linea del archivo de simulaciones planificadas
    s_p = read_yaml_file('simulaciones_planificadas.yaml')

    for i in range(len(s_p)):
        update_yaml_file(
            'parameters.yaml',
            s_p[i]['x'],
            s_p[i]['y'],
            s_p[i]['planner'],
            s_p[i]['mpc'],
            s_p[i]['UF'],
            s_p[i]['sigma_max'],
            s_p[i]['map']
        )

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        name = './matlab/' + timestamp + '_simulacion_' + str(i)

        # 为当前这次仿真构造一个 CSV 路径
        result_csv = os.path.join(
            args.result_dir,
            f"{args.planner_type}_{timestamp}_sim_{i}.csv"
        )

        # Tarea 1: Ejecutar prepare_simulation.py
        print('##################### PREPARE SIMULATION  1/6 ##################### N: ' + str(i))
        p = run_command("python prepare_simulation.py")
        p.wait()

        print('##################### LAUNCH SIMULATION  2/6 ##################### N: ' + str(i))
        # Tarea 2: Ejecutar roslaunch test.launch
        ros_process = run_command("roslaunch test.launch")

        # Esperar 20 segundos
        time.sleep(20)

        print('##################### LAUNCH ACTIVE SLAM 3/6 ##################### N: ' + str(i))
        # Tarea 3: Ejecutar waypoint manager y active_slam_core_rrt.py en procesos paralelos
        waypoint = run_command("python way_point_manager.py")

        # 把 planner_type 和 result_path 传给 active_slam_core_rrt.py
        active_cmd = (
            f"python active_slam_core_rrt.py "
            f"--planner_type {args.planner_type} "
            f"--result_path {result_csv}"
        )
        active_slam_process = run_command(active_cmd)

        # Esperar a que el proceso active_slam_core_rrt.py termine
        active_slam_process.wait(timeout=2500)
        waypoint.terminate()
        # active_slam_process.terminate()  # normalmente不需要

        print('##################### ACTIVE SLAM FINALIZED 4/6 ##################### N: ' + str(i))

        # Tarea 4: Ejecutar python save_map.py
        p = run_command("python save_map.py " + name)
        p.wait(timeout=120)
        p.terminate()
        print('##################### DATA SAVED 5/6 ##################### N: ' + str(i))

        # Tarea 5: Cerrar la simulación
        p = subprocess.call(['rosnode', 'kill', '--all'])
        print('##################### SIMULATION CLOSED 6/6 ##################### N: ' + str(i))

        time.sleep(60)

