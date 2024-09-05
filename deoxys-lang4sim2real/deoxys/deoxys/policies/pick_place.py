import numpy as np


class PickPlace:

    def __init__(self, env, pick_height_thresh=0.1, xyz_action_scale=15.0,
                 pick_pt_noise=0.0, drop_pt_noise=0.0,
                 gripper_ac=-0.5):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.z_std = 0.01
        self.xyz_action_scale = xyz_action_scale
        self.pick_pt_noise = pick_pt_noise
        self.drop_pt_noise = drop_pt_noise
        self.gripper_ac = gripper_ac
        self.reset(np.zeros(3,), np.zeros(3,))
        self.post_grasp_num_wait_ts = 2

    def reset(
            self, pick_pt, drop_pt, target_obj_name="",
            gripper_ac=None, lift_pt_z=None):
        self.pick_pt = pick_pt
        self.drop_pt = drop_pt

        self.pick_height_thresh_noisy = (
            self.pick_height_thresh + np.random.normal(scale=self.z_std))

        self.drop_pt = drop_pt

        self.lift_pt = self.set_lift_pt(lift_pt_z)

        print("self.pick_pt", self.pick_pt)
        print("self.lift_pt", self.lift_pt)
        print("self.drop_pt", self.drop_pt)

        self.grasp_attempted = False
        self.lift_attempted = False
        self.place_attempted = False

        self.post_grasp_ts = 0

        self.target_obj = target_obj_name

        if gripper_ac is not None:
            self.gripper_ac = gripper_ac

    def set_lift_pt(self, lift_pt_z):
        # Set a lift point which is in the xy midpoint of the pick_pt
        # and drop_pt
        lift_pt = 0.5 * (self.pick_pt + self.drop_pt)
        if lift_pt_z is None:
            lift_pt_z = self.drop_pt[2]
        lift_pt[2] = lift_pt_z
        return lift_pt

    def get_action(self, obs):
        ee_pos = obs["ee_pos"]
        ee_pickpt_dist = np.linalg.norm(self.pick_pt - ee_pos)
        ee_liftpt_dist = np.linalg.norm(self.lift_pt - ee_pos)
        ee_droppt_dist = np.linalg.norm(self.drop_pt - ee_pos)
        action_xyz = [0., 0., 0.]
        action_angles = [0., 0.]
        action_gripper = [-1.]
        done = False
        err_thresh = 0.03
        stage_num = -1

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            stage_num = 6
            pass
            # print('post place zero action')
        elif ee_pickpt_dist > err_thresh and not self.grasp_attempted:
            # move near the object
            stage_num = 1
            action_xyz = (self.pick_pt - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > err_thresh:
                stage_num = 0
                action_xyz[2] = 0.0
            # print('move near object')
        elif not self.grasp_attempted:
            stage_num = 2
            # near the object enough, performs grasping action
            action_xyz = (self.pick_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
            self.grasp_attempted = True
        elif self.post_grasp_ts <= self.post_grasp_num_wait_ts:
            stage_num = 2
            action_gripper = [self.gripper_ac]
        elif ee_liftpt_dist > err_thresh and not self.lift_attempted:
            # lifting objects above the height threshold for picking
            stage_num = 3
            action_xyz = (self.lift_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
        elif ee_droppt_dist > err_thresh:
            # lifted, now need to move towards the container
            stage_num = 4
            self.lift_attempted = True
            action_xyz = (self.drop_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff < 2 * err_thresh:
                stage_num = 5
        else:
            # already moved above the container; drop object
            stage_num = 6
            action_gripper = [-1.]
            self.place_attempted = True

        if self.grasp_attempted:
            self.post_grasp_ts += 1

        agent_info = dict(
            grasp_attempted=self.grasp_attempted,
            place_attempted=self.place_attempted,
            done=done,
            policy_lang_stage_num=stage_num,
            total_num_stages=7,
        )

        # clip action
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper))
        action = np.clip(action, -1, 1)

        return action, agent_info


class PickPlaceN(PickPlace):
    def __init__(self, *args, **kwargs):
        super().__init__(xyz_action_scale=15.0, *args, **kwargs)

    def set_lift_pt(self, lift_pt_z):
        # Set a lift point which is right above the pick_pt
        lift_pt = np.array(self.pick_pt)
        if lift_pt_z is None:
            lift_pt_z = 0.18 + self.env.gripper_z_offset
        lift_pt[2] = lift_pt_z
        return lift_pt

    def get_action(self, obs):
        ee_pos = obs["ee_pos"]
        ee_pickpt_dist = np.linalg.norm(self.pick_pt - ee_pos)
        ee_liftpt_dist = np.linalg.norm(self.lift_pt - ee_pos)
        ee_droppt_dist = np.linalg.norm(self.drop_pt - ee_pos)
        ee_droppt_xy_dist = np.linalg.norm(self.drop_pt[:2] - ee_pos[:2])
        action_xyz = [0., 0., 0.]
        action_angles = [0., 0.]
        action_gripper = [-1.]
        done = False
        err_thresh = 0.03
        stage_num = -1

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            stage_num = 6
            pass
            # print('post place zero action')
        elif ee_pickpt_dist > err_thresh and not self.grasp_attempted:
            # move near the object
            stage_num = 1
            action_xyz = (self.pick_pt - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > err_thresh:
                stage_num = 0
                action_xyz[2] = 0.0
            # print('move near object')
        elif not self.grasp_attempted:
            stage_num = 2
            # near the object enough, performs grasping action
            action_xyz = (self.pick_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
            self.grasp_attempted = True
        elif self.post_grasp_ts <= self.post_grasp_num_wait_ts:
            stage_num = 2
            action_gripper = [self.gripper_ac]
        elif ee_liftpt_dist > err_thresh and not self.lift_attempted:
            # lifting objects above the height threshold for picking
            stage_num = 3
            action_xyz = (self.lift_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
        elif ee_droppt_xy_dist > err_thresh:
            # lifted, now need to move xy-direction towards a point above
            # container
            stage_num = 4
            self.lift_attempted = True
            action_xyz = (self.drop_pt - ee_pos) * self.xyz_action_scale
            action_xyz[2] = 0.0  # don't move down from lift point.
            action_gripper = [self.gripper_ac]
        elif ee_droppt_dist > err_thresh:
            # Above container, but need to match the drop_pt_z
            stage_num = 5
            action_xyz = (self.drop_pt - ee_pos) * self.xyz_action_scale
            action_gripper = [self.gripper_ac]
        else:
            # already moved above the container; drop object
            stage_num = 6
            action_gripper = [-1.]
            self.place_attempted = True

        if self.grasp_attempted:
            self.post_grasp_ts += 1

        agent_info = dict(
            grasp_attempted=self.grasp_attempted,
            place_attempted=self.place_attempted,
            done=done,
            policy_lang_stage_num=stage_num,
            total_num_stages=7,
        )

        # clip action
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper))
        action = np.clip(action, -1, 1)

        return action, agent_info
