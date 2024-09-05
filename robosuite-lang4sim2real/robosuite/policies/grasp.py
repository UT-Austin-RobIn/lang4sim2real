import numpy as np
from robosuite.policies import Policy


class GraspPolicy(Policy):
    def __init__(
            self, eef_pos_obs_key, target_obj_obs_key, pick_z_thresh,
            pos_sensitivity=1.0, rot_sensitivity=1.0, env=None):
        self._reset_internal_state()
        self._reset_state = 0

        self.eef_pos_obs_key = eef_pos_obs_key
        self.target_obj_obs_key = target_obj_obs_key
        self.target_obj = None
        self.pick_z_thresh = pick_z_thresh
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.object_offsets = {None: np.zeros(3), "milk": np.array([0,0,0.045]), "bread": np.zeros(3), "can": np.array([0,0,0.02]), "cereal": np.array([0,0,0.05])}

    def _reset_internal_state(self):
        self.rotation = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False
        self.num_timesteps_after_grasp = 0

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
        target_obj_pos = obs[self.target_obj_obs_key]
        target_obj_lifted = target_obj_pos[2] > self.pick_z_thresh
        gripper_to_target_obj_dist = np.linalg.norm(target_obj_pos - ee_pos)
        gripper_to_target_obj_xy_dist = np.linalg.norm(target_obj_pos[:2] - ee_pos[:2])
        obj_offset = self.object_offsets[self.target_obj]
        # print("obs", obs)

        action_xyz = np.zeros(3)
        if gripper_to_target_obj_xy_dist > 0.01 and not self.grasp:
            # Move above object
            action_xyz = target_obj_pos - ee_pos
            action_xyz[2] = 0.0
        elif gripper_to_target_obj_dist > 0.01 + np.linalg.norm(obj_offset):
            # Move near object
            action_xyz = target_obj_pos - ee_pos + obj_offset
        elif not self.grasp:
            # Close gripper
            self.grasp = True
            action_xyz = target_obj_pos - ee_pos
        elif self.num_timesteps_after_grasp < 10:
            pass
        elif not target_obj_lifted:
            # Lift object above grasp threshold
            action_xyz = np.array([0., 0., .1])
        else:
            # zero actions
            pass

        if self.grasp:
            self.num_timesteps_after_grasp += 1

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
