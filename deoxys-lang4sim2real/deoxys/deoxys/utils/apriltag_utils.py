from dt_apriltags import Detector
import numpy as np

import deoxys.utils.calibration_utils as cal_utils
from deoxys.utils.params import *

class AprilTagDetector:
    def __init__(self):
        self.at_detector = Detector(searchpath=['apriltags'],
                               families='tag36h11',
                               nthreads=1,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0)
        self.camera_intrinsic_params = [635.64732655, 635.64732655, 64, 64] # fx fy kx ky
        # camera_intrinsic_params=[635.64732655, 635.64732655, 321.29969964, 241.43618241]
        self.tag_size = 0.04

    def get_tag_pixel_corners(self, img):
        tags = self.at_detector.detect(
            img, estimate_tag_pose=True,
            camera_params=self.camera_intrinsic_params,
            tag_size=self.tag_size)

        assert len(tags) == 1, "Did not find at least one tag"
        # tags[0].tag_id may be useful.
        return tags[0].corners

    def get_tag_rotation(self, img):
        corners = self.get_tag_pixel_corners(img)
        bottom_xy_diff = corners[1] - corners[0]
        bottom_angle = -bottom_xy_diff[1] / bottom_xy_diff[0]
        # Y needs to be negated b/c image top row of pixels has
        # lowest y value.
        top_xy_diff = corners[2] - corners[3]
        top_angle = -top_xy_diff[1] / top_xy_diff[0]
        theta_est = np.arctan([bottom_angle, top_angle])
        # print("theta_est", theta_est)

        if abs(theta_est[1] - theta_est[0]) > 3:
            theta = theta_est[0]
        else:
            theta = np.mean(theta_est)

        # Determine which quadrant we are in
        in_quad_2_3 = 0.5 * (
            (corners[0][0] - corners[1][0]) +
            (corners[3][0] - corners[2][0])) > 0.0
        in_quad_1_2 = -0.5 * (
            (corners[1][1] - corners[0][1]) +
            (corners[2][1] - corners[3][1])) > 0.0
        in_quad_2 = in_quad_2_3 and in_quad_1_2
        in_quad_3 = in_quad_2_3 and not in_quad_1_2
        # print("in_quad_2", in_quad_2)
        # print("in_quad_3", in_quad_3)
        if in_quad_2:
            theta += np.pi
        elif in_quad_3:
            theta -= np.pi
        return theta

    def get_robot_xy_of_tag(self, img):
        four_corners = self.get_tag_pixel_corners(img)
        corners = four_corners.mean(axis=0)
        robot_xy_coords = cal_utils.rgb_to_robot_coords(
            corners, AT_RGB_TO_ROBOT_TRANSMATRIX)
        return robot_xy_coords