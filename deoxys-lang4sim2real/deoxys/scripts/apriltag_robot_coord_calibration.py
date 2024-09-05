from deoxys.envs import init_env
import deoxys.utils.calibration_utils as cal_utils
from deoxys.utils.apriltag_utils import AprilTagDetector
from deoxys.utils.params import *

import argparse
import cv2
import numpy as np
from tqdm import tqdm


def collect_robot_and_rgb_coords(env):
    obs = env.reset()
    pick_pt_z = 0.06
    o = np.array([0.3, -.48, pick_pt_z])
    dx, dy = np.array([0.17, 0, 0]), np.array([0, 0.2, 0])
    # dx, dy = np.array([0, 0, 0]), np.array([0, 0, 0])
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
    rgb_corners = [] # april tag corners

    atd = AprilTagDetector()
    for pt in tqdm(pts):
        ee_pos = env.reach_xyz_pos(pt, max_num_tries=12)
        robot_ee_pos_coords.append(ee_pos)

        input("Press ENTER to calibrate the point")
        env.reset()

        print("getting image")
        img = env.get_obs()['image_480x640']
        # convert to grayscale for april tag
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("img.shape", img.shape)
        
        corners = atd.get_tag_pixel_corners(img)
        # corners = corners.reshape(-1,) # (x1,y1,x2,y2,x3,y3,x4,y4)
        corners = corners.mean(axis=0) # avg all 4 corners to get center point
        rgb_corners.append(corners)

        print("robot_ee_pos_coords", robot_ee_pos_coords)
        print("rgb_corners", rgb_corners)

    return np.array(robot_ee_pos_coords)[:, :2], np.array(rgb_corners)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--obj-id", type=int, default=0)
    args = parser.parse_args()

    env = init_env("frka_base")
    rgb_coords = np.array([
        np.array([219.96118927, 130.29620552]),
        np.array([347.67125702, 135.50828934]),
        np.array([477.48858643, 143.04146767]),
        np.array([480.74164581, 242.14374161]),
        np.array([346.29714203, 241.46017838]),
        np.array([222.81478119, 238.58164215]),
        np.array([226.42630005, 346.47872162]),
        np.array([350.03766632, 350.83940125]),
        np.array([479.22989655, 346.21283722]),
    ])
    robot_coords = np.array([
        np.array([ 0.29710683, -0.45112626,  0.05238249]),
        np.array([ 0.30420942, -0.26301037,  0.06279326]),
        np.array([ 0.32429188, -0.07886463,  0.06754538]),
        np.array([ 0.46522906, -0.06885675,  0.0601992 ]),
        np.array([ 0.46170059, -0.26419046,  0.05486927]),
        np.array([ 0.45455116, -0.4477161 ,  0.04717154]),
        np.array([ 0.60802975, -0.44033758,  0.03937203]),
        np.array([ 0.6210971 , -0.2618538 ,  0.04938713]),
        np.array([ 0.61808434, -0.07297846,  0.05178878]),
    ])[:, :2]
    print("robot_coords.shape", robot_coords.shape)
    print("rgb_coords.shape", rgb_coords.shape)
    trfm_matrix = cal_utils.compute_transform(robot_coords, rgb_coords)
    print(trfm_matrix.shape)
