import argparse

import numpy as np

from deoxys.envs.franka_env import FrankaEnv
from deoxys.utils.control_utils import reset_joints
from deoxys.utils.polygon_utils import PolygonWrapper
from rlkit.lang4sim2real_utils.lang_templates import (
    ww_lang_by_stages, ww_lang_by_stages_v2)


class FrankaWireWrap(FrankaEnv):
    def __init__(
            self, realrobot_access=True, target_obj_name="",
            obj_set=0, stage_num_scheme="v1",
            pause_in_violation_box=True, **kwargs):
        self.flex_end_obj_name = target_obj_name  # "plug"
        self.target_obj_name = target_obj_name
        # ^ thing that robot tries to grasp
        if obj_set == 0:
            self.flex_obj_name = "cord"  # only used for language
            self.center_obj_name = "blender"
        elif obj_set == 1:
            self.flex_obj_name = "ethernet cable"  # only used for language
            self.center_obj_name = "3d_printer_spool"
        else:
            raise NotImplementedError
        self.center_pt_xy_offset = np.array([0.05, 0.0])
        self.flex_marker_name = "rb_ckbd"

        self.pause_in_violation_box = pause_in_violation_box

        super().__init__(realrobot_access=realrobot_access, **kwargs)

        # wider workspace bounds
        self.workspace_xyz_limits = {
            "lo": np.array([0.28, -0.24, 0.025 + self.gripper_z_offset]),
            "hi": np.array([0.65, 0.22, 0.3 + self.gripper_z_offset]),
        }

        # rough bounds of the object on scene
        neg_pad = 0.03
        self.wrapped_obj_init_limits = {
            True: {  # gripper opened
                "lo": np.array(
                    [0.3367 + neg_pad,
                     -0.1698 + neg_pad,
                     self.workspace_xyz_limits["lo"][2]]),
                "hi": np.array([
                    0.5559,
                    0.15 - neg_pad,
                    0.248 - 0.5 * neg_pad])},
            False: {  # gripper closed
                "lo": np.array([
                    0.3367 + neg_pad,
                    -0.1276 + neg_pad,
                    self.workspace_xyz_limits["lo"][2]]),
                "hi": np.array([
                    0.5559 - neg_pad, 0.122 - neg_pad, 0.248])},
        }
        min_x = (0.8 * self.workspace_xyz_limits["lo"][0]
                 + 0.2 * self.workspace_xyz_limits["hi"][0])
        min_y = self.workspace_xyz_limits["lo"][1] + 0.03
        max_x = (0.5 * self.wrapped_obj_init_limits[True]["hi"][0]
                 + 0.5 * self.wrapped_obj_init_limits[True]["lo"][0])
        max_y = self.wrapped_obj_init_limits[True]["lo"][1] - 0.02
        self.obj_xy_init_polygon = PolygonWrapper([
            (max_x, min_y),
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
        ])

        # reward related terms
        self.wrap_ang_success_thresh = (5 / 3) * np.pi
        self.flex_marker_to_center_obj_dist_thresh = 0.05

        self.flex_end_ang_change_hist_across_resets = []

        if realrobot_access:
            self.obj_detector = self.obj_detector_dl_cls(
                env=self, skip_reset=True)

        assert stage_num_scheme in [
            "v1",  # portion rotated
            "v2",  # position relative to obj
        ]
        self.stage_num_scheme = stage_num_scheme

    def get_gripper_type(self):
        return 1

    def reset(self, lift_before_reset=False, do_forward=True):
        self.do_forward = do_forward
        # Add to logging across resets
        if hasattr(self, "flex_end_ang_change_hist"):
            self.flex_end_ang_change_hist_across_resets.append(
                np.sum(self.flex_end_ang_change_hist))
        # if lift_before_reset:
        ee_pos_z = self.get_obs()["ee_pos"][2]
        reset_z = 0.27
        print("lifting before reset")
        while ee_pos_z < 0.27 - 0.03:
            self.flex_end_ang_change_hist = []
            self.flex_end_xy_hist = []
            self.step(
                np.array([0, 0, 15.0 * (reset_z - ee_pos_z), 0, 0, -1]),
                calc_reward=False, ignore_eepos_limits=True)
            ee_pos_z = self.get_obs()["ee_pos"][2]
        reset_joints(
            self.robot_interface,
            self.type_to_controller_cfg_map["joint"], self.logger,
            reset_joint_pos_idx=2)
        print("done resetting.")
        obs = self.get_obs()
        self.flex_end_ang_change_hist = []
        self.flex_end_xy_hist = []
        init_ang_info_dict = self.get_flex_end_ang_info()
        self.log_angles(init_ang_info_dict)
        try:
            # there's a 3cm offset between where the obj detector thinks
            # the blender center is and where it actually is.
            # Shift it up more because it keeps getting stuck in the back from
            # joint issues and not enough clearance in the front.
            center_pt = self.get_obj_xy_pos_fn(self, self.center_obj_name)
            if self.center_obj_name == "blender":
                center_pt += self.center_pt_xy_offset
            print("center_pt", center_pt)
        except KeyError:
            center_pt = np.array([0.46, 0.0])
            # print(f"failed to update center pos. assuming {center_pt}")
        self.center_pos = center_pt
        return obs

    def is_do_forward(self):
        # This is called before the reset in data_collection.py
        # so we need to add in the current self.flex_end_ang_change_hist
        ang_change_sum = np.abs(np.sum(
            self.flex_end_ang_change_hist_across_resets
            + self.flex_end_ang_change_hist))
        print("ang_change_sum", ang_change_sum)
        return ang_change_sum <= (np.pi / 2)

    def propose_obj_xy_init_pos(self):
        return self.obj_xy_init_polygon.sample_unif_rand_pt()

    def check_obj_xy_valid_init_pos(self, obj_xy_pos):
        return self.obj_xy_init_polygon.contains(obj_xy_pos)

    def get_center_pos(self):
        return self.center_pos

    def get_flex_end_ang_info(self):
        try:
            obj_pt = self.get_obj_xy_pos_fn(self, self.flex_end_obj_name)
            center_pt = self.get_center_pos()
        except:
            print(
                f"Failed to find objects: {self.flex_end_obj_name}, "
                f"{self.center_obj_name}")
            return {}

        x, y = obj_pt - center_pt
        ang = np.arctan2(y, x)
        if ang < 0:
            ang += 2 * np.pi  # ang in [0, 2pi]
        xy = np.array([x, y])
        if len(self.flex_end_xy_hist) == 0:
            ang_change = 0.0
        else:
            ang_change = self.get_ang_change(xy, self.flex_end_xy_hist[-1])
        return dict(abs_ang=ang, xy=xy, ang_change=ang_change)

    def get_ang_change(self, xy1, xy0):
        """
        First build a rotation matrix for xy0
        Then "unrotate" both xy0 and xy1 by same amt so xy0 = [1, 0]
        Returns a value in [-pi, pi]
        """
        x1, y1 = xy1
        x0, y0 = xy0
        theta = np.arctan2(y0, x0)
        inv_rot_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])
        xy1_unrotated_by_xy0_ang = inv_rot_mat @ xy1
        ang_change = np.arctan2(
            xy1_unrotated_by_xy0_ang[1], xy1_unrotated_by_xy0_ang[0])
        # ^ in [-pi, pi]
        # R^T = R^{-1}, since orthogonal
        return ang_change  # in [-pi, pi]

    def log_angles(self, ang_info_dict):
        if ang_info_dict:
            self.flex_end_ang_change_hist.append(ang_info_dict['ang_change'])
            self.flex_end_xy_hist.append(ang_info_dict['xy'])

    def get_wrapped_angle(self):
        ang_info_dict = self.get_flex_end_ang_info()
        # print("ang_info_dict", ang_info_dict)
        self.log_angles(ang_info_dict)

        # Sum up angular distance traveled
        integral_estimate = np.sum(self.flex_end_ang_change_hist)

        if ang_info_dict:
            atan2_estimate = self.get_ang_change(
                ang_info_dict["xy"], self.flex_end_xy_hist[0])  # in [-pi, pi]

            # combine the estimates
            # I believe this works for integral_estimates <= -pi as well.
            num_2pis = (integral_estimate + np.pi) // (2 * np.pi)
            estimate = (num_2pis * 2 * np.pi) + atan2_estimate
        else:
            estimate = integral_estimate

        db_str = f"estimate {estimate} integral_estimate {integral_estimate}"
        try:
            db_str += f" atan2_estimate {atan2_estimate}"
        except:
            pass
        if np.random.random() < 0.2:
            print(db_str)

        return estimate

    def get_reward(self, info):
        if self.do_forward:
            # ccw
            wrapped_enough = (
                self.get_wrapped_angle() >= self.wrap_ang_success_thresh)
        else:
            # cw. In reality this var should be called "unwrapped_enough"
            wrapped_enough = (
                self.get_wrapped_angle() <= -self.wrap_ang_success_thresh)
        success = (
            wrapped_enough
            and self.is_gripper_open()
            and self.flex_not_on_center_obj())
        return float(success)

    def flex_not_on_center_obj(self):
        try:
            flex_pts = self.get_obj_xy_pos_fn(
                self, self.flex_marker_name, conf_thresh=0.5)  # (n, 2)
        except KeyError:
            flex_pts = []
        center_obj_pt = self.get_center_pos()  # (2,)
        for flex_pt in flex_pts:
            if (np.linalg.norm(flex_pt - center_obj_pt)
                    < self.flex_marker_to_center_obj_dist_thresh):
                return False
        return True

    def eepos_in_limits(self, obs):
        def in_box(pt, lo_hi_dict):
            within_lo = np.all(pt >= lo_hi_dict["lo"])
            within_hi = np.all(pt <= lo_hi_dict["hi"])
            return within_lo and within_hi

        within_workspace = in_box(obs['ee_pos'], self.workspace_xyz_limits)
        within_bad_box = in_box(
            obs['ee_pos'],
            self.wrapped_obj_init_limits[self.is_gripper_open()])
        if within_bad_box and self.pause_in_violation_box:
            import ipdb; ipdb.set_trace()
        return within_workspace and not within_bad_box

    def get_lang_by_stages(self):
        # the language list is the same for forward and backward.
        if self.stage_num_scheme == "v1":
            lang_by_stage_fn = ww_lang_by_stages
        elif self.stage_num_scheme == "v2":
            lang_by_stage_fn = ww_lang_by_stages_v2
        else:
            raise NotImplementedError
        langs_by_stage = lang_by_stage_fn(
            self.target_obj_name,
            self.flex_obj_name,
            self.center_obj_name)
        return langs_by_stage

    def get_task_lang_dict(self):
        """
        Task instruction language (used for multitask language-conditioned BC)
        Index in list is the task idx
        If doing sim2real, should match the sim languages.
        """
        # return [""] * self.num_tasks
        flex_obj_name = self.flex_obj_name.replace("_", " ")
        center_obj_name = self.center_obj_name.replace("_", " ")
        task_lang_list = [
            f"Wrap the {flex_obj_name} counterclockwise around the {center_obj_name}",
            f"Unwrap the {flex_obj_name} clockwise around the {center_obj_name}",
        ]
        task_lang_dict = dict(
            instructs=task_lang_list,
        )
        return task_lang_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=int, default=0)
    args = parser.parse_args()

    env = FrankaWireWrap()
    obs = env.reset()
    for t in range(args.ts):
        env.get_wrapped_angle()
        obs = env.step([0., 0., 0., 0., 0., -1])
