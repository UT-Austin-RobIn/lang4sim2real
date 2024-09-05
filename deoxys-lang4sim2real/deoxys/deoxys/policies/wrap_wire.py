import numpy as np


class WrapWire:
    def __init__(
            self, env, gripper_ac=-0.5, clockwise=False, xyz_action_scale=15.0,
            stage_num_scheme="v1",
            lifted_thresh=0.13, desired_dist_to_center=0.15):
        self.env = env
        self.gripper_ac = gripper_ac
        self.reset(np.zeros(3,), np.zeros(3,))
        self.post_grasp_num_wait_ts = 2
        self.clockwise = clockwise
        self.lifted_thresh = lifted_thresh
        self.desired_dist_to_center = desired_dist_to_center
        self.xyz_action_scale = xyz_action_scale
        assert stage_num_scheme in [
            "v1",  # portion rotated
            "v2",  # position relative to obj
        ]
        self.stage_num_scheme = stage_num_scheme
        self.total_num_stages = len(self.env.get_lang_by_stages())

    def reset(self, pick_pt, drop_pt, clockwise=False, gripper_ac=None):
        self.pick_pt = pick_pt
        self.grasp_attempted = False
        self.drop_pt = drop_pt
        self.place_attempted = False
        self.achieved_done = False
        self.post_grasp_ts = 0
        self.clockwise = clockwise
        if gripper_ac is not None:
            self.gripper_ac = gripper_ac
        self.last_stage_num = 0  # store the previous stage num for lang
        self.env.reset(do_forward=not clockwise)

    def get_wrap_xyzg_action(self, obs):
        action_gripper = [self.gripper_ac]

        rel_pos = np.zeros(3,)
        rel_pos[:2] = obs["ee_pos"][:2] - self.env.get_center_pos()

        if self.stage_num_scheme == "v1":
            angle_abs = np.clip(
                np.abs(self.env.get_wrapped_angle()), 0, 2 * np.pi)
            if self.clockwise:  # unwrap
                # should be from 9-13
                stage_num = 13 - int(((angle_abs + np.pi/4) // (np.pi/2)))
            else:  # wrap
                # should be from 4-8
                stage_num = 4 + int(((angle_abs + np.pi/4) // (np.pi/2)))
        elif self.stage_num_scheme == "v2":
            # swap x and y
            gripper_center_ang_plus_offset = (
                np.arctan2(rel_pos[0], rel_pos[1]) + (np.pi / 4))
            # gripper_center_ang_plus_offset is from [-3pi/4, 5pi/4]
            if gripper_center_ang_plus_offset < 0:
                gripper_center_ang_plus_offset += 2 * np.pi
            # gripper_center_ang_plus_offset is from [0, 2pi]
            if self.clockwise:  # unwrap
                stage_num = (
                    8 + int(gripper_center_ang_plus_offset // (np.pi / 2)))
            else:  # wrap
                stage_num = (
                    4 + int(gripper_center_ang_plus_offset // (np.pi / 2)))
        else:
            raise NotImplementedError
        # print(stage_num)

        dist_to_center = np.linalg.norm(rel_pos)
        normalized = (rel_pos / dist_to_center) * self.desired_dist_to_center
        action_xyz = 10 * rel_pos * (
            self.desired_dist_to_center - dist_to_center)
        # ^ move toward/away from center
        tangent_movement = 1 * np.array(
            [-normalized[1], normalized[0], 0.0])  # Move tangent to circle
        if self.clockwise:
            action_xyz -= tangent_movement
        else:
            action_xyz += tangent_movement
        action_xyz[2] = (
            (0.05 + self.lifted_thresh) - obs["ee_pos"][2])
        # ^ move up/down to maintain height
        return action_xyz, action_gripper, stage_num, rel_pos

    def get_action(self, obs):
        ee_pos = obs["ee_pos"]

        ccw_done_ang_thresh = self.env.wrap_ang_success_thresh + (np.pi / 6)
        if self.clockwise:  # unwrap
            done_thresh = -ccw_done_ang_thresh
            self.done = self.env.get_wrapped_angle() <= done_thresh
        else:
            self.done = self.env.get_wrapped_angle() >= ccw_done_ang_thresh
            self.half_done = (
                self.env.get_wrapped_angle() >= 0.5 * ccw_done_ang_thresh)
        action_xyz = np.zeros(3)
        action_angles = [0., 0.]
        action_gripper = [-1]
        pick_xyz_dist = np.linalg.norm(self.pick_pt - ee_pos)
        drop_xyz_dist = np.linalg.norm(self.drop_pt - ee_pos)

        if self.place_attempted:
            # do nothing
            stage_num = (
                (self.total_num_stages - 2) if not self.clockwise
                else (self.total_num_stages - 1))
        elif pick_xyz_dist > 0.03 and not self.grasp_attempted:
            action_xyz = self.pick_pt - ee_pos
            xy_dist = np.linalg.norm(action_xyz[:2])
            if xy_dist > 0.02:
                stage_num = 0
                action_xyz[2] = 0.0
                # ^ don't move down when you're still far from the object
                # beause you'll hit stuff
            elif pick_xyz_dist > 0.03:
                stage_num = 1
            else:
                stage_num = 2
                action_gripper = [self.gripper_ac]
                self.grasp_attempted = True
        elif self.post_grasp_ts <= self.post_grasp_num_wait_ts:
            stage_num = 2
            action_gripper = [self.gripper_ac]
            self.grasp_attempted = True
        elif not self.achieved_done and ee_pos[2] < self.lifted_thresh:
            stage_num = 3
            action_gripper = [self.gripper_ac]
            action_xyz = np.array(
                [0., 0., 0.05 + self.lifted_thresh - ee_pos[2]])
        elif not self.done and not self.achieved_done:
            action_xyz, action_gripper, stage_num, rel_pos = (
                self.get_wrap_xyzg_action(obs))
        elif not self.place_attempted and drop_xyz_dist > 0.03:
            # move to drop_pt, keep gripper closed
            stage_num = self.last_stage_num
            action_xyz = self.drop_pt - ee_pos
            action_gripper = [self.gripper_ac]
            self.achieved_done = True
            # self.grasp_attempted = False
        else:
            # open gripper
            stage_num = self.last_stage_num
            self.place_attempted = True

        if self.grasp_attempted:
            self.post_grasp_ts += 1

        agent_info = dict(
            grasp_attempted=self.grasp_attempted,
            place_attempted=self.place_attempted,
            done=self.achieved_done and self.place_attempted,
            policy_lang_stage_num=stage_num,
            total_num_stages=self.total_num_stages,
        )
        self.last_stage_num = stage_num

        action = np.concatenate(
            (action_xyz * self.xyz_action_scale, action_angles, action_gripper)
        )
        if np.any(np.abs(action[:2]) >= 1):
            action[:2] /= np.max(np.abs(action[:2]))
        action = np.clip(action, -1, 1)
        print(action, stage_num)

        return action, agent_info
