from deoxys.envs import init_env
import deoxys.utils.calibration_utils as cal_utils
from deoxys.utils.obj_detector import ObjectDetectorDL
from deoxys.utils.params import *

import argparse
import numpy as np
from tqdm import tqdm


def collect_robot_and_rgb_coords(env, obj_detector, obj_id):
    obj_name = obj_detector.classes[obj_id]
    print(f"Using {obj_name} to calibrate")
    obs = env.reset()
    pick_pt_z = OBJ_TO_PICK_PT_Z_MAP[obj_name]
    o = np.array([0.34, -.15, pick_pt_z])
    dx, dy = np.array([0.12, 0, 0]), np.array([0, 0.12, 0])
    pts = np.array([
        o,
        o + dy,
        o + 2*dy,
        o + dx + 2*dy,
        o + dx + dy,
        o + dx,
        o + 2*dx,
        o + 2*dx + dy,
        o + 2*dx + 2*dy,
    ])

    robot_ee_pos_coords = []
    rgb_coords = []
    for pt in tqdm(pts):
        ee_pos = env.reach_xyz_pos(pt)
        robot_ee_pos_coords.append(ee_pos)

        input("Press ENTER to calibrate the point")

        img = obj_detector.get_img(transpose=False)
        print("img.shape", img.shape)
        centroids = obj_detector.get_centroids(img)
        print("centroids", centroids)
        rgb_coords.append(centroids[obj_name])

    return robot_ee_pos_coords, rgb_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-id", type=int, default=0)
    args = parser.parse_args()

    env = init_env("frka_base")
    obj_detector = ObjectDetectorDL(env=env)
    robot_coords, rgb_coords = collect_robot_and_rgb_coords(
        env, obj_detector, args.obj_id)
    # robot_coords, rgb_coords = np.random.rand(9, 2), np.random.rand(9, 2)
    robot_coords = np.array(robot_coords)[:, :2]
    rgb_coords = np.array(rgb_coords)[:, :2]
    trfm_matrix = cal_utils.compute_transform(robot_coords, rgb_coords)
