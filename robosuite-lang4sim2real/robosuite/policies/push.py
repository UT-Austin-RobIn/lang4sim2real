import numpy as np
from robosuite.policies import Policy
from robosuite.utils.object_utils import OBJ_NAME_TO_XYZ_OFFSET

class PushPolicy(Policy):
    def __init__(
            self, eef_pos_obs_key, target_obj_obs_key, target_dst_obs_key, pos_sensitivity=1.0, rot_sensitivity=1.0, pick_z_thresh=None, env=None):
        self._reset_internal_state()
        self._reset_state = 0

        self.eef_pos_obs_key = eef_pos_obs_key
        self.target_obj_obs_key = target_obj_obs_key
        self.target_dst_obs_key = target_dst_obs_key
        self.target_obj = None
        self.pick_z_thresh = pick_z_thresh
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def _reset_internal_state(self):
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = True
        self.num_timesteps_after_grasp = 0
        self.behind = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        # self._reset_state = 0
        # self._enabled = True

    def get_action(self, obs):
        ee_pos = obs[self.eef_pos_obs_key]
        obj_offset = OBJ_NAME_TO_XYZ_OFFSET[self.target_obj]
        target_obj_pos = obs[self.target_obj_obs_key] - np.array([0, 0, obj_offset[2]]) # target pushing the base of the object.
        target_dst_pos = obs[self.target_dst_obs_key]

        action_xyz = np.zeros(3)
        # go behind object and push

        unit_vector_scale = 0.07
        push_vector = target_dst_pos - target_obj_pos
        target_obj_to_target_dst_dist = np.linalg.norm(push_vector)
        neg_unit_vector = push_vector / -np.linalg.norm(push_vector)
        neg_unit_vector[2] = 0
        neg_unit_vector = neg_unit_vector * unit_vector_scale 

        behind_obj_pos = neg_unit_vector + target_obj_pos
        gripper_to_behind_obj_xy_dist = np.linalg.norm(ee_pos[:2] - behind_obj_pos[:2])
        gripper_to_behind_obj_dist = np.linalg.norm(ee_pos - behind_obj_pos)


        if gripper_to_behind_obj_xy_dist > 0.1 and not self.behind:
            # Move behind object relative to destination
            action_xyz = behind_obj_pos - ee_pos
            action_xyz[2] = 0.0
        elif gripper_to_behind_obj_dist > 0.01 and not self.behind:
            # Move near object
            action_xyz = behind_obj_pos - ee_pos
        elif target_obj_to_target_dst_dist > 0.03: 
            self.behind = True
            gripper_to_obj_xy_dist = np.linalg.norm(ee_pos[:2] - target_obj_pos[:2])
            if ee_pos[2] > 1:
                self.behind = False
            elif gripper_to_obj_xy_dist > unit_vector_scale + 0.05:
                action_xyz[2] = 0.5
            else:
                action_xyz = push_vector * min(1 // np.linalg.norm(push_vector), 1)
                action_xyz[2] = 0
        else:
            # zero actions
            pass
        ##TODO reset push state if first push is not enough and make this better
        dpos = self.pos_sensitivity * action_xyz # self.pos - self.last_pos
        # self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )
