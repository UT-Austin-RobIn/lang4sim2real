import argparse
import time

import cv2
import numpy as np

from deoxys.utils.params import *
from deoxys.envs.franka_env import FrankaEnv
import deoxys.utils.calibration_utils as cal_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["obj", "at"], default="obj")
    args = parser.parse_args()

    if args.mode == "obj":
        from deoxys.utils.obj_detector import ObjectDetectorDL
    elif args.mode == "at":
        from deoxys.utils.apriltag_utils import AprilTagDetector

    env = FrankaEnv()

    while True:
        obs = env.reset()
        time.sleep(1)

        if args.mode == "obj":
            obj_detector = ObjectDetectorDL(env=env)
            img = obj_detector.get_img(transpose=False)
            centroids = obj_detector.get_centroids(img)
            obj_name = "paper_rect_box"
            robot_xy_coords = cal_utils.rgb_to_robot_coords(
                centroids[obj_name], DL_RGB_TO_ROBOT_TRANSMATRIX)
            pick_pt_z = OBJ_TO_PICK_PT_Z_MAP[obj_name]
        elif args.mode == "at":
            atd = AprilTagDetector()
            img = env.get_obs()['image_480x640']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            four_corners = atd.get_tag_pixel_corners(img)
            corners = four_corners.mean(axis=0)

            theta = atd.get_tag_rotation(img)
            robot_xy_coords = atd.get_robot_xy_of_tag(img)
            pick_pt_z = 0.07

        print("robot_xy_coords", robot_xy_coords)
        env.reach_xyz_pos(
            np.concatenate((robot_xy_coords, [pick_pt_z]), axis=0),
            max_num_tries=10, err_tol=0.02)
        time.sleep(2)
