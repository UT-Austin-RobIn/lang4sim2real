import numpy as np

from rlkit.lang4sim2real_utils.lang_templates import (
    ww_lang_by_stages, ww_lang_by_stages_simplified)
from robosuite.policies import Policy
from robosuite.utils.object_utils import OBJ_NAME_TO_XYZ_OFFSET


class WrapPolicy(Policy):
    def __init__(
            self, eef_pos_obs_key, target_geom_obs_key,
            pick_z_thresh, center_geom_key, relative_location_lang,
            pos_sensitivity=1.0, env=None, drop_pos_offset=None):
        self.relative_location_lang = relative_location_lang
        self.pos_sensitivity = pos_sensitivity
        self._reset_state = 0
        self.eef_pos_obs_key = eef_pos_obs_key
        self.target_geom_obs_key = target_geom_obs_key
        self.grasp = False
        self.pick_z_thresh = pick_z_thresh
        self.center_geom_key = center_geom_key
        self.num_timesteps_after_grasp = 0
        self.target_obj = "spoolandwire"
        self.env = env

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        # self._reset_state = 0
        # self._enabled = True

    def _reset_internal_state(
            self, pick_point_z=None, drop_point=None, object_to_target=None):
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)
        # ^ immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False 
        self.place_attempted = False
        self.stack_ready = False
        self.num_timesteps_after_grasp = 0
        self.last_stage_num = 0

    def get_action(self, obs):
        ee_pos = obs[self.eef_pos_obs_key]
        target_geom_pos = obs[self.target_geom_obs_key]
        center_pos = obs[self.center_geom_key]
        obj_offset = OBJ_NAME_TO_XYZ_OFFSET[self.target_obj]
        target_geom_pos_w_grasp_offset = target_geom_pos + obj_offset
        obj_lifted = target_geom_pos[2] > self.pick_z_thresh 
        gripper_to_target_obj_xy_dist = np.linalg.norm(
            target_geom_pos_w_grasp_offset[:2] - ee_pos[:2])
        clockwise = self.env.task_id == 1 or self.env.task_id == 3
        if clockwise:
            self.done = self.env.reward(
                threshold=-self.env.success_wrap_ang - (1/6) * np.pi)
        else:
            self.done = self.env.reward(
                threshold=self.env.success_wrap_ang + (1/6) * np.pi)
        action_xyz = np.zeros(3)
        if self.place_attempted:
            # do nothing
            if self.relative_location_lang:
                stage_num = 12
            else:
                stage_num = 14
            pass
        elif not self.grasp:
            action_xyz = 2 * (target_geom_pos_w_grasp_offset - ee_pos)
            if (gripper_to_target_obj_xy_dist > 0.02
                    or abs(action_xyz[2]) > 0.01):
                stage_num = 1
                if gripper_to_target_obj_xy_dist > .02:
                    stage_num = 0
                    if ee_pos[2] < .97:
                        action_xyz[2] = 0.0
                        # ^ don't move down when still far from the object
                        # beause you'll hit stuff
            else:
                # Close gripper
                stage_num = 2
                self.num_timesteps_after_grasp = 0
                self.grasp = True
        elif self.num_timesteps_after_grasp < 12:
            # wait some time while gripper closes
            # This probably breaks markovianness of the policy.
            stage_num = 2
            self.num_timesteps_after_grasp += 1
        elif not obj_lifted:
            # Lift object above pick&place threshold
            stage_num = 3
            action_xyz = np.array(
                [0., 0., 0.05 + self.pick_z_thresh - target_geom_pos[2]])
        elif not self.done:
            # wrap
            rel_pos = ee_pos - center_pos

            if not self.relative_location_lang:
                angle = self.env.get_angle_wrapped()
                stage_num = 4 + int(
                    (np.abs(angle) + (np.pi / 4)) // (np.pi / 2))
                if clockwise:
                    stage_num += 5
            else:
                angle = np.arctan2(rel_pos[0], -rel_pos[1])
                angle += np.pi / 4
                if angle < 0:
                    angle += 2 * np.pi
                elif angle > 2 * np.pi:
                    angle -= 2 * np.pi
                stage_num = 4 + int(angle // (np.pi / 2))
                if clockwise:
                    stage_num += 4

            rel_pos[2] = 0.0
            desired_dist_to_center = 0.022
            dist_to_center = np.linalg.norm(rel_pos)
            normalized = (rel_pos / dist_to_center) * desired_dist_to_center
            action_xyz = 3 * rel_pos * (
                desired_dist_to_center - dist_to_center)
            # ^ move toward/away from center
            tangent_movement = 12 * np.array(
                [-normalized[1], normalized[0], 0.0])
            # ^ Move tangent to circle
            if clockwise:
                action_xyz -= tangent_movement
            else:
                action_xyz += tangent_movement
            action_xyz[2] = (0.05 + self.pick_z_thresh) - target_geom_pos[2]
            # ^ move up/down to maintain height
        else:
            # place
            stage_num = self.last_stage_num
            self.grasp = False
            self.place_attempted = True
            pass
        dpos = self.pos_sensitivity * action_xyz  # self.pos - self.last_pos
        # self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)

        action_dict = dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

        if self.relative_location_lang:
            lang_list = ww_lang_by_stages_simplified
        else:
            lang_list = ww_lang_by_stages
        info_dict = dict(
            policy_lang_list=lang_list(),
            policy_lang_stage_num=stage_num,
            total_num_stages=len(lang_list()),
        )

        self.last_stage_num = stage_num

        return action_dict, info_dict
