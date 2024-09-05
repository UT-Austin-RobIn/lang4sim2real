import numpy as np

from robosuite.policies import Policy
from robosuite.utils.object_utils import OBJ_NAME_TO_XYZ_OFFSET
from rlkit.lang4sim2real_utils.lang_templates import pp_lang_by_stages_v1


class PickPlacePolicy(Policy):
    def __init__(
            self, eef_pos_obs_key, target_obj_obs_key,
            pick_z_thresh, pos_sensitivity=1.0, env=None, stack=False,
            stack_obj_obs_key="platform_pos", drop_pos_offset=None,
            mm_kwargs={}):
        self._reset_internal_state()
        self._reset_state = 0

        self.eef_pos_obs_key = eef_pos_obs_key
        self.target_obj_obs_key = target_obj_obs_key
        self.target_obj = None
        self.pick_z_thresh = pick_z_thresh

        self.pos_sensitivity = (
            mm_kwargs.get("action_coeff", 1.0) * pos_sensitivity)
        self.env = env

        self.stack = stack
        self.stack_obj_obs_key = stack_obj_obs_key
        if drop_pos_offset is not None:
            self.drop_pos_offset = drop_pos_offset
        else:
            self.drop_pos_offset = np.zeros(3,)

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()

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

    def get_action(self, obs):
        ee_pos = obs[self.eef_pos_obs_key]
        obj_offset = OBJ_NAME_TO_XYZ_OFFSET[self.target_obj]
        target_obj_pos = obs[self.target_obj_obs_key]
        target_obj_pos_w_grasp_offset = target_obj_pos + obj_offset
        obj_lifted = target_obj_pos[2] > self.pick_z_thresh
        gripper_to_target_obj_dist = np.linalg.norm(
            target_obj_pos_w_grasp_offset - ee_pos)
        gripper_to_target_obj_xy_dist = np.linalg.norm(
            target_obj_pos_w_grasp_offset[:2] - ee_pos[:2])

        drop_pos = None
        cont_name = None
        if self.stack:
            cont_name = self.stack_obj_obs_key.split("_")[0]
            # Raise obj high enough for cont
            drop_pos = obs[self.stack_obj_obs_key]
        else:
            drop_pos = self.env.target_bin_placements[self.env.object_id]
        drop_pos += self.drop_pos_offset

        gripper_droppos_xy_dist = np.linalg.norm(
            drop_pos[:2] - target_obj_pos[:2])
        gripper_droppos_z_dist = ee_pos[2] - drop_pos[2]
        # print(gripper_droppos_xy_dist)
        stage_num = -1

        if cont_name in ["platform", None]:
            # For backward consistency reasons
            cont_name = "container"
        lang_by_stages = pp_lang_by_stages_v1(self.target_obj, cont_name)

        action_xyz = np.zeros(3)
        if self.place_attempted:
            stage_num = 6
            pass
        elif not self.grasp:
            action_xyz = target_obj_pos_w_grasp_offset - ee_pos
            if gripper_to_target_obj_xy_dist > 0.01:
                stage_num = 0
                action_xyz[2] = 0.0
            elif gripper_to_target_obj_dist > 0.01:
                # Move near object
                stage_num = 1
            else:
                # Close gripper
                stage_num = 2
                self.grasp = True
        elif self.num_timesteps_after_grasp < 10:
            # wait some time while gripper closes
            # This probably breaks markovianness of the policy.
            stage_num = 2
            pass
        elif ((not obj_lifted)
              and (gripper_droppos_xy_dist > 0.02)
              and (not self.stack_ready)):
            # Lift object above pick&place threshold
            stage_num = 3
            action_xyz = np.array(
                [0., 0., 0.02 + self.pick_z_thresh - target_obj_pos[2]]
            ) + obj_offset
        elif gripper_droppos_xy_dist > 0.02:
            # print("going to droppos", gripper_droppos_xy_dist)
            stage_num = 4
            action_xyz = drop_pos - target_obj_pos
            action_xyz[2] = 0.0
        elif self.stack and gripper_droppos_z_dist - obj_offset[2] > 0.09:
            stage_num = 5
            self.stack_ready = True
            action_xyz = drop_pos - ee_pos + (2 * obj_offset)
        else:
            # print("tried to place it")
            stage_num = 6
            self.grasp = False
            self.place_attempted = True

        if self.grasp:
            self.num_timesteps_after_grasp += 1

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

        info_dict = dict(
            policy_lang_list=lang_by_stages,
            policy_lang_stage_num=stage_num,
            total_num_stages=7,
        )

        assert (
            info_dict["policy_lang_stage_num"] < info_dict["total_num_stages"])

        return action_dict, info_dict
