import argparse
import os
import time

import cv2
import gym
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from deoxys import config_root
from deoxys.utils import params, transform_utils, YamlConfig
from deoxys.utils.control_utils import reset_joints
from deoxys.utils.log_utils import get_deoxys_example_logger
from rlkit.lang4sim2real_utils.lang_templates import (
    pp_lang_by_stages_v1,
    pp_lang_by_stages_v1_rev
)


class FrankaEnv(gym.Env):
    def __init__(
            self, realrobot_access=True, state_mode=0, substeps_per_step=1):
        assert state_mode in {0, 1}
        self.state_mode = state_mode
        if state_mode == 0:
            self.gripper_state_idx = 3
        elif state_mode == 1:
            self.gripper_state_idx = -1
        self.substeps_per_step = substeps_per_step

        interface_cfg = "charmander.yml"
        if interface_cfg[0] != "/":
            self.config_path = os.path.join(config_root, interface_cfg)
        else:
            self.config_path = interface_cfg
        print("self.config_path", self.config_path)
        if realrobot_access:
            self.import_realrobot_pkgs()

        self.type_to_controller_cfg_map = {
            "position": YamlConfig(
                    os.path.join(config_root, "osc-position-controller.yml")
                ).as_easydict(),
            "joint": YamlConfig(
                    os.path.join(config_root, "joint-position-controller.yml")
                ).as_easydict()
        }

        self.controller_type = "OSC_POSITION"
        # self.controller_type = "OSC_YAW"

        # Define action_space
        if self.controller_type == "OSC_POSITION":
            self.action_dim = 6
            act_bound = 1
            act_high = np.ones(self.action_dim) * act_bound
            self.action_space = gym.spaces.Box(-act_high, act_high)
            self.gripper_idx = self.action_dim - 1   # needed for railrl-ut
        elif self.controller_type == "OSC_YAW":
            self.action_dim = 7
            act_bound = 1
            act_high = np.ones(self.action_dim) * act_bound
            self.action_space = gym.spaces.Box(-act_high, act_high)
            self.gripper_idx = self.action_dim - 1
        else:
            raise NotImplementedError

        self.observation_space = self.create_obs_space()

        self.gripper_type = self.get_gripper_type()
        # ^ 0 = orig gripper; 1 = long new gripper
        assert self.gripper_type in {0, 1}

        if self.gripper_type == 1:
            self.gripper_z_offset = 0.08
        else:
            self.gripper_z_offset = 0.0

        self.workspace_xyz_limits = {
            "lo": np.array([0.28, -0.17, 0.025 + self.gripper_z_offset]),
            "hi": np.array([0.59, 0.20, 0.3 + self.gripper_z_offset]),
        }

        self.num_steps = 1

        # Controller freq
        self.freq = 2  # Hz

    def get_gripper_type(self):
        return 0

    def import_realrobot_pkgs(self):
        """
        Keep this standalone so that when we run the training script,
        we don't need to import these things.
        """
        from deoxys.franka_interface import FrankaInterface
        from rpl_vision_utils.networking.camera_redis_interface import (
            CameraRedisSubInterface)
        self.robot_interface = FrankaInterface(self.config_path)

        from deoxys.utils.data_collection_utils import get_objs_bbox
        self.get_objs_bbox_fn = get_objs_bbox
        from deoxys.utils.obj_detector import ObjectDetectorDL
        self.obj_detector_dl_cls = ObjectDetectorDL

        from deoxys.utils.data_collection_utils import get_obj_xy_pos
        self.get_obj_xy_pos_fn = get_obj_xy_pos
        self.cr_interface = CameraRedisSubInterface(
            redis_host="172.16.0.1", camera_id=0)
        self.cr_interface.start()

        self.logger = get_deoxys_example_logger()

    def create_obs_space(self):
        def create_obs_box(dim):
            obs_bound = 100
            obs_high = np.ones(dim) * obs_bound
            obs_space = gym.spaces.Box(-obs_high, obs_high)
            return obs_space

        self.im_shape = (128, 128, 3)
        assert self.im_shape[0] == self.im_shape[1]
        self.image_length = np.prod(self.im_shape)
        img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                   dtype=np.float32)
        if self.state_mode == 0:
            spaces = {
                'image': img_space,
                'ee_states': create_obs_box(16),  # (16,)
                'joint_pos': create_obs_box(7),  # (7,)
                'gripper_states': create_obs_box(1),  # (1,)
                'ee_pos': create_obs_box(3),  # (3,)
                'state': create_obs_box(16 + 7 + 1 + 3),
            }
        elif self.state_mode == 1:
            spaces = {
                'image': img_space,
                'ee_states': create_obs_box(16),  # (16,)
                'joint_pos': create_obs_box(7),  # (7,)
                'ee_pos': create_obs_box(3),  # (3,)
                'ee_quat': create_obs_box(4),  # (4,)
                'joint_pos_cos': create_obs_box(7),  # (7,)
                'joint_pos_sin': create_obs_box(7),  # (7,)
                'gripper_states': create_obs_box(1),  # (1,)
                'state': create_obs_box(3 + 4 + 7 + 7 + 1),
            }
        else:
            raise NotImplementedError

        observation_space = gym.spaces.Dict(spaces)
        return observation_space

    def reset(self, lift_before_reset=False, reset_joint_pos_idx=None):
        if reset_joint_pos_idx is None:
            reset_joint_pos_idx = self.gripper_type
        if lift_before_reset:
            print("lifting before reset")
            self.step(np.array([0, 0, 0.6, 0, 0, -1]))
        reset_joints(
            self.robot_interface,
            self.type_to_controller_cfg_map["joint"], self.logger,
            reset_joint_pos_idx=reset_joint_pos_idx)
        print("done resetting.")
        return self.get_obs()

    def eepos_in_limits(self, obs):
        within_lo = np.all(obs['ee_pos'] >= self.workspace_xyz_limits['lo'])
        within_hi = np.all(obs['ee_pos'] <= self.workspace_xyz_limits['hi'])
        return within_lo and within_hi

    def step(
            self, action, time_overhead=0.0, calc_reward=True,
            ignore_eepos_limits=False):
        action = np.clip(action, -1, 1)
        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action,
            controller_cfg=self.type_to_controller_cfg_map["position"],
        )
        info = self.get_info()
        info["forced_reset"] = False
        if calc_reward:
            r = self.get_reward(info)
        else:
            r = 0.0
        done = False

        if time_overhead > 1 / self.freq:
            # Error case
            print("Overhead time bigger than 1 / freq.")
            time_overhead = 0.0

        secs_to_sleep = (1 / self.freq) - time_overhead
        # ^ Needed to wait for robot to finish action
        all_substep_obs_dict = self.sleep_with_img_logging(secs_to_sleep)
        # To get an accurate get_obs()
        # For now, this is a hard-coded delay.
        obs = self.get_obs()
        if not ignore_eepos_limits and not self.eepos_in_limits(obs):
            print("violating workspace safety limits. Resetting.")
            # Alternatively we can take the reverse action (-action).
            self.reset()
            info["forced_reset"] = True
        obs = append_obs_dicts(all_substep_obs_dict, obs)

        return obs, r, done, info

    def sleep_with_img_logging(self, secs_to_sleep):
        """
        Collects k + 1 = (self.substeps_per_step - 1) obs readings
        Returns obs_dict with len-k list for each obs_key.
        """
        all_substep_obs_dict = {}
        k = self.substeps_per_step
        for i in range(k):
            time.sleep(secs_to_sleep / k)
            if i < k - 1:
                # Save obs if not the final iteration
                # the obs for final iteration is captured by `step`
                substep_obs_dict = self.get_obs()
                all_substep_obs_dict = append_obs_dicts(
                    all_substep_obs_dict, substep_obs_dict)
        return all_substep_obs_dict

    def reach_xyz_pos(self, targ_pos, max_num_tries=6, err_tol=0.02):
        # Tries to reach a target xyz position `targ_pos`
        assert targ_pos.shape == (3,)
        action_mult = 15.0
        ee_pos = self.get_obs_state()['ee_pos']

        err = np.linalg.norm(targ_pos - ee_pos)
        num_tries = 0
        while err > err_tol and num_tries <= max_num_tries:
            action_xyz = action_mult * (targ_pos - ee_pos)
            print("action_xyz", action_xyz)
            action = np.concatenate([action_xyz, [0., 0., -1.]], axis=0)
            print("action", action)
            nobs, _, _, info = self.step(action)
            if info["forced_reset"]:
                break
            ee_pos = self.get_obs_state()['ee_pos']
            print("ee_pos", ee_pos)
            err = np.linalg.norm(targ_pos - ee_pos)
            print("err", err)
            # print("nobs['joint_states']", nobs['joint_states'])
            num_tries += 1

        return ee_pos

    def reach_joint_pose(self, joint_pose):
        assert len(joint_pose) == 7
        action = np.concatenate([joint_pose, [-1.0]], axis=0)
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                self.logger.info(
                    f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
                self.logger.info(
                    f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")

                if (
                    np.max(
                        np.abs(
                            np.array(self.robot_interface._state_buffer[-1].q)
                            - np.array(joint_pose)
                        )
                    )
                    < 1e-3
                ):
                    break
            self.robot_interface.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=self.type_to_controller_cfg_map["joint"],
            )

    def center_crop(self, im_obs):
        side_len = min(im_obs.shape[:2])
        if im_obs.shape[0] == im_obs.shape[1]:
            return im_obs
        
        half_side_len = int(side_len / 2)
        center_x = im_obs.shape[0] // 2
        center_y = im_obs.shape[1] // 2
        center_cropped = im_obs[
            (center_x - half_side_len):(center_x + half_side_len),
            (center_y - half_side_len):(center_y + half_side_len)]
        return center_cropped

    def get_obs_state(self):
        last_state = self.robot_interface._state_buffer[-1]
        last_gripper_q = self.robot_interface.last_gripper_q

        if last_gripper_q is None:
            print("forcing last_gripper q to be -1")
            last_gripper_q = np.array(-1.0)

        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),  # (16,)
            "joint_pos": np.array(last_state.q),  # (7,)
            "joint_pos_cos": np.cos(last_state.q),  # (7,)
            "joint_pos_sin": np.sin(last_state.q),  # (7,)
            "gripper_states": last_gripper_q[None],  # (1, )
        }

        # _, ee_pos = self.robot_interface.last_eef_quat_and_pos()
        O_T_EE = np.array(last_state.O_T_EE).reshape(4, 4).transpose()
        ee_pos = O_T_EE[:3, 3:].flatten()
        ee_rot = O_T_EE[:3, :3]
        ee_quat = transform_utils.mat2quat(ee_rot)
        # print("ee_pos", ee_pos)
        state_dict['ee_pos'] = ee_pos  # (3,)
        state_dict['ee_quat'] = ee_quat  # (4,)

        return state_dict

    def get_obs(self):
        obs_dict = {}

        # Get Image
        im_info = self.cr_interface.get_img()
        obs_dict['image_480x640'] = cv2.cvtColor(
            im_info["color"], cv2.COLOR_RGB2BGR)
        im_obs = self.center_crop(obs_dict['image_480x640'])
        im_obs = cv2.resize(im_obs, (self.im_shape[1], self.im_shape[0]))
        obs_dict['image'] = im_obs

        # Get State
        state_dict = self.get_obs_state()
        obs_dict.update(state_dict)

        # Have a single entry in obs_dict that concats everything else
        if self.state_mode == 0:
            state_keys = [
                "ee_pos", "gripper_states", "ee_states", "joint_pos"]
        elif self.state_mode == 1:
            state_keys = [
                "ee_pos", "ee_quat", "joint_pos_cos", "joint_pos_sin",
                "gripper_states"]
        else:
            raise NotImplementedError

        obs_list = [obs_dict[k] for k in state_keys]
        obs_dict["state"] = np.concatenate(obs_list)
        # print("obs_dict['state']", obs_dict['state'])

        return obs_dict

    def gripper_open(self, gripper_state_vec):
        """Returns bool vector (or scalar) given state_vec of same dim."""
        return gripper_state_vec >= 0.07

    def is_gripper_open(self, print_state=False):
        state_dict = self.get_obs_state()
        gripper_open = self.gripper_open(state_dict['gripper_states'])
        if np.random.random() < 0.2 or print_state:
            print(
                f"state_dict['gripper_states'] {state_dict['gripper_states']} "
                f"gripper_open: {gripper_open}")
        return gripper_open.all()

    def get_reward(self, info):
        return 0

    def get_info(self):
        return {}

    def render_obs(self):
        pass

    def _set_action_space(self):
        pass

    def _set_obs_space(self):
        pass


class FrankaEnvPP(FrankaEnv):
    """PickPlace env for FrankaEnv."""
    def __init__(self, realrobot_access=True, target_obj_name="", **kwargs):
        self.target_obj_name = target_obj_name
        super().__init__(realrobot_access=realrobot_access, **kwargs)
        # self.obj_detector = ObjectDetectorDL(env=self)

        # Possible starting obj locations in robot xy coords

        self.workspace_xyz_limits = {
            "lo": np.array([0.3, -0.17, 0.025 + self.gripper_z_offset]),
            "hi": np.array([0.59, 0.12, 0.3 + self.gripper_z_offset]),
        }

        pad = 0.03
        loc_interp = 0.6
        self.obj_xy_init_limits = {
            "lo": self.workspace_xyz_limits["lo"][:2] + pad,
            "hi": np.array([
                self.workspace_xyz_limits["hi"][0] - pad,
                (loc_interp * self.workspace_xyz_limits["lo"][1] +
                    (1 - loc_interp) * self.workspace_xyz_limits["hi"][1]),
            ]),
        }

        if realrobot_access:
            self.obj_detector = self.obj_detector_dl_cls(
                env=self, skip_reset=True)

    def obj_xy_placed_in_cont(self, obj_xy):
        xy_sat = (
            (params.CONT_XY_LIMITS["lo"] <= obj_xy).all()
            and (obj_xy <= params.CONT_XY_LIMITS["hi"]).all())
        gripper_open = self.gripper_open(
            self.get_obs_state()['gripper_states'])
        return xy_sat and gripper_open

    def obj_xy_placed_out_cont(self, obj_xy):
        xy_sat = (
            (params.CONT_XY_LIMITS["lo"] <= obj_xy).all()
            and (obj_xy <= params.CONT_XY_LIMITS["hi"]).all())
        gripper_open = self.gripper_open(
            self.get_obs_state()['gripper_states'])
        return (not xy_sat) and gripper_open

    def propose_obj_xy_init_pos(self):
        return np.random.uniform(
            low=self.obj_xy_init_limits["lo"],
            high=self.obj_xy_init_limits["hi"])

    def check_obj_xy_valid_init_pos(self, obj_xy_pos):
        above_lo = (self.obj_xy_init_limits["lo"] <= obj_xy_pos).all()
        below_hi = (obj_xy_pos <= self.obj_xy_init_limits["hi"]).all()
        if not above_lo or not below_hi:
            print(f"{obj_xy_pos} not in bounds ({self.obj_xy_init_limits['lo']}, {self.obj_xy_init_limits['hi']})")
        return above_lo and below_hi

    def get_lang_by_stages(self, do_forward, obj_name):
        if do_forward:
            lang_by_stage_fn = pp_lang_by_stages_v1
        else:
            lang_by_stage_fn = pp_lang_by_stages_v1_rev
        langs_by_stage = lang_by_stage_fn(obj_name)
        return langs_by_stage

    def get_task_lang_dict(self):
        """
        Task instruction language (used for multitask language-conditioned BC)
        Index in list is the task idx
        If doing sim2real, should match the sim languages.
        """
        # return [""] * self.num_tasks
        task_lang_list = [
            f"stack {self.target_obj_name} in bin",
            f"stack {self.target_obj_name} out of bin",
        ]
        task_lang_dict = dict(
            instructs=task_lang_list,
        )
        return task_lang_dict


class FrankaEnvObjToBowlToPlate(FrankaEnvPP):
    def __init__(self, *args, **kwargs):
        assert kwargs["target_obj_name"] != ""
        self.step_kwargs = dict(
            reward_modes=["pp_in", "pp_in"],
            skills=["pick_place_n", "pick_place_n"],  # used for scripted policy
            verbs=["stack", "stack"],  # used for task instruction language
            obj_names=[kwargs["target_obj_name"], "clear_container"],
            cont_names=["clear_container", "plate"],
            drop_pos_offsets=[
                # carrot
                np.array([0.0, 0.01, 0.07]),
                np.array([-0.01, -0.05, 0.08]),
                # bridge
                # np.array([0.0, 0.02, 0.07]),
                # np.array([0.01, -0.04, 0.08]),
            ],
        )
        self.rev_step_kwargs = dict(
            # list items in order of rev step idx
            pick_pt_z_offsets=[
                0.03, 0.0],
            lift_pt_z_offsets=[
                # carrot
                0.05, 0.04],
                # bridge
                # 0.07, 0.04],
            drop_pos_offsets=[
                np.array([0.0, 0.0, 0.05]),
                np.array([0.0, 0.0, 0.02]),
            ],
        )
        self.step_idx = 0
        self.rev_step_idx = 2

        self.waited_ts_forward = 0
        self.waited_ts_backward = 0
        # num timesteps to do zero actions when a step has been finished
        # before advancing to next step
        self.num_wait_ts_between_steps = 2

        super().__init__(*args, **kwargs)

        self.num_steps = len(self.step_kwargs['reward_modes'])

        wlim_x_lo, wlim_y_lo = self.workspace_xyz_limits["lo"][:2]
        wlim_x_hi, wlim_y_hi = self.workspace_xyz_limits["hi"][:2]
        self.init_placement_distr = {
            f"{self.target_obj_name}": dict(
                low=self.obj_xy_init_limits["lo"],
                high=np.array([
                    self.obj_xy_init_limits["hi"][0],
                    0.7 * wlim_y_lo + 0.3 * wlim_y_hi])),
            "clear_container": dict(
                low=np.array([
                    0.8 * wlim_x_lo + 0.2 * wlim_x_hi,
                    0.4 * wlim_y_lo + 0.6 * wlim_y_hi]),
                high=np.array([
                    0.75 * wlim_x_lo + 0.25 * wlim_x_hi,
                    0.3 * wlim_y_lo + 0.7 * wlim_y_hi]),)
        }

    def get_gripper_type(self):
        return 1

    def get_task_lang_dict(self):
        """
        Task instruction language (used for multitask language-conditioned BC)
        Index in list is the task idx
        If doing sim2real, should match the sim languages.
        """
        def get_multistep_instruction(do_forward):
            step_instrs = []
            full_instr = ""
            for step_idx in range(self.num_steps):
                verb_str = self.step_kwargs["verbs"][step_idx]
                obj_str = self.step_kwargs["obj_names"][step_idx].replace(
                    "_", " ")
                cont_str = self.step_kwargs["cont_names"][step_idx].replace(
                    "_", " ")
                do_forward_to_template_instruction_map = {
                    True: f"{verb_str} {obj_str} on {cont_str}",
                    False: f"{verb_str} {obj_str} out of {cont_str}",
                }
                step_instr = do_forward_to_template_instruction_map[do_forward]
                step_instrs.append(step_instr)

            if not do_forward:
                step_instrs = step_instrs[::-1]
            full_instr = ", then ".join(step_instrs)
            return full_instr

        task_lang_list = [
            get_multistep_instruction(do_forward=True),
            get_multistep_instruction(do_forward=False),
        ]
        task_lang_dict = dict(
            instructs=task_lang_list,
        )
        return task_lang_dict

    def reset(self, do_forward=True, lift_before_reset=False):
        obs = super().reset(lift_before_reset=lift_before_reset)
        self.do_forward = do_forward
        self.step_idx = 0
        self.rev_step_idx = self.get_rev_step_idx()
        return obs

    def get_reward(self, info, rews_by_step=None):
        # action is not actually being used by this function or its calls
        if rews_by_step is None:
            rews_by_step = self.reward_by_step(info)
        return float((np.array(rews_by_step) == 1.0).all())

    def get_reward_by_mode(self, reward_mode, obj_name, cont_name):
        # We assume that the obj is smaller than the container.
        if reward_mode in ["pp_in", "not_pp_in"]:
            # if not self.is_gripper_open():
            #     return 0.0
            bboxes = self.get_objs_bbox_fn(
                self, lift_before_reset=False, fmt="vertex_list",
                wait_secs=0.001)
            try:
                obj_bbox = bboxes[obj_name]
                cont_bbox = bboxes[cont_name]
            except:
                print("Failed to find objects")
                return 0.0
            obj_poly = Polygon(obj_bbox)
            cont_poly = Polygon(cont_bbox)
            intersection = obj_poly.intersection(cont_poly).area
            ioo = intersection / obj_poly.area  # intersection over object
            # print(f"ioo({obj_name}, {cont_name})", ioo)
            if reward_mode == "pp_in":
                success = ioo > 0.5
            elif reward_mode == "not_pp_in":
                success = ioo < 0.1
            return float(success)
        elif reward_mode in ["pp_on", "not_pp_on"]:
            # if not self.is_gripper_open():
            #     return 0.0
            bboxes = self.get_objs_bbox_fn(
                self, lift_before_reset=False, fmt="xyxy",
                wait_secs=0.001)
            try:
                obj_bbox = bboxes[obj_name]
                cont_bbox = bboxes[cont_name]
            except:
                print("Failed to find objects")
                return 0.0
            x_pts = [
                (obj_bbox[0], 0),
                (obj_bbox[2], 0),
                (cont_bbox[0], 1),
                (cont_bbox[2], 1),
            ]
            sorted_x_pts, ids = list(zip(*sorted(x_pts)))
            if (ids[0] == ids[1]) and (ids[2] == ids[3]):
                # [0, 0, 1, 1] or [1, 1, 0, 0]
                x_ioo = 0.0
            else:
                # [0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]
                # Take middle two points.
                x_intersection = sorted_x_pts[2] - sorted_x_pts[1]
                obj_pixel_width = obj_bbox[2] - obj_bbox[0]
                x_ioo = x_intersection / obj_pixel_width
            obj_y_min = obj_bbox[1]
            obj_y_max = obj_bbox[3]
            obj_pixel_height = obj_y_max - obj_y_min
            cont_y_min = cont_bbox[1]
            cont_y_max = cont_bbox[3]
            obj_y_above_cont_y = (
                (obj_y_min >= cont_y_min)
                and ((obj_y_min - cont_y_max) <= obj_pixel_height))
            if reward_mode == "pp_on":
                success = (x_ioo > 0.5) and (obj_y_above_cont_y)
            elif reward_mode == "not_pp_on":
                success = (x_ioo < 0.1) or (not obj_y_above_cont_y)
            return float(success)

    def get_rev_step_idx(self, rev_rews_by_step=None):
        if rev_rews_by_step is None:
            rev_rews_by_step = self.rev_reward_by_step(info={})
        rev_step_idx = (rev_rews_by_step + [0]).index(0)
        return rev_step_idx

    def get_step_idx(self, rews_by_step=None):
        if rews_by_step is None:
            rews_by_step = self.reward_by_step(info={})
        step_idx = (rews_by_step + [0]).index(0)
        return step_idx

    def rev_reward_by_step(self, info, check_step_idx_grasp=False):
        rev_rews_by_step = []
        for rev_step_idx in range(self.num_steps):
            step_idx = self.num_steps - rev_step_idx - 1
            # rev_step_idx --> step_idx
            # 0 --> 2 - 0 - 1 = 1
            # 1 --> 2 - 1 - 1 = 0
            reward_mode = "not_" + self.step_kwargs['reward_modes'][step_idx]
            obj_name = self.step_kwargs["obj_names"][step_idx]
            cont_name = self.step_kwargs["cont_names"][step_idx]
            rev_step_rew = self.get_reward_by_mode(
                reward_mode, obj_name, cont_name)
            rev_rews_by_step.append(rev_step_rew)
        if check_step_idx_grasp:
            rev_rews_by_step = self.maybe_modify_step_rew_list(
                rev_rews_by_step, self.rev_step_idx)
        assert set(rev_rews_by_step).issubset({0.0, 1.0})
        return rev_rews_by_step

    def reward_by_step(self, info, check_step_idx_grasp=False):
        """
        if check_step_idx_grasp == True, check grasp opened for
        current step_idx.
        """
        rews_by_step = []
        for step_idx in range(self.num_steps):
            reward_mode = self.step_kwargs['reward_modes'][step_idx]

            obj_name = self.step_kwargs["obj_names"][step_idx]
            cont_name = self.step_kwargs["cont_names"][step_idx]

            step_rew = self.get_reward_by_mode(
                reward_mode, obj_name, cont_name)
            rews_by_step.append(step_rew)
        if check_step_idx_grasp:
            rews_by_step = self.maybe_modify_step_rew_list(
                rews_by_step, self.step_idx)
        assert set(rews_by_step).issubset({0.0, 1.0})
        return rews_by_step

    def maybe_modify_step_rew_list(
            self, step_rew_list, idx, gripper_open=None):
        # Only mark current rew by step as success if gripper is open
        if idx == len(step_rew_list):
            return step_rew_list
        if gripper_open is None:
            gripper_open = self.is_gripper_open()
        step_rew_list[idx] = float(
            step_rew_list[idx] and gripper_open)
        return step_rew_list

    def step(self, action, time_overhead=0.0):
        obs, r, done, info = super().step(
            action, time_overhead, calc_reward=False)

        # override reward
        rews_by_step = self.reward_by_step(info, check_step_idx_grasp=True)
        rev_rews_by_step = self.rev_reward_by_step(
            info, check_step_idx_grasp=True)

        if self.do_forward:
            print("rews_by_step", rews_by_step)
            r = self.get_reward(info, rews_by_step)
        else:
            print("rev_rews_by_step", rev_rews_by_step)
            r = self.get_reward(info, rev_rews_by_step)

        # Increment step_idx if succeeded current step
        if (self.step_idx <= (self.num_steps - 1)
                and rews_by_step[self.step_idx] == 1):
            if self.waited_ts_forward >= self.num_wait_ts_between_steps:
                self.step_idx = self.get_step_idx(rews_by_step)
                self.waited_ts_forward = 0
            else:
                self.waited_ts_forward += 1
            print("self.step_idx", self.step_idx)
            print("self.waited_ts_forward", self.waited_ts_forward)

        # Update rev_step_idx if succeeded at current step
        if (self.rev_step_idx <= (self.num_steps - 1)
                and rev_rews_by_step[self.rev_step_idx] == 1):
            # rev_step_idx can jump by more than 1 if
            # a previous step was already completed
            if self.waited_ts_backward >= self.num_wait_ts_between_steps:
                self.rev_step_idx = self.get_rev_step_idx(rev_rews_by_step)
                self.waited_ts_backward = 0
            else:
                self.waited_ts_backward += 1
            print("self.rev_step_idx", self.rev_step_idx)
            print("self.waited_ts_backward", self.waited_ts_backward)

        # Store multistep info
        info.update(dict(
            rews_by_step=rews_by_step,
            rev_rews_by_step=rev_rews_by_step,
            step_idx=self.step_idx,
            rev_step_idx=self.rev_step_idx,
            do_forward=int(self.do_forward),
        ))

        return obs, r, done, info

    def propose_xy_init_pos(self, obj_name):
        return np.random.uniform(**self.init_placement_distr[obj_name])

    def get_obj_cont_name(self, do_forward=True, step_idx=None):
        if step_idx is None:
            if do_forward:
                step_idx = self.step_idx
            else:  # backward
                step_idx = self.num_steps - self.rev_step_idx - 1

        if step_idx == self.num_steps:
            target_obj_name = ""
            cont_name = ""
        else:
            target_obj_name = self.step_kwargs['obj_names'][step_idx]
            cont_name = self.step_kwargs['cont_names'][step_idx]
        return target_obj_name, cont_name

    def get_lang_by_stages(self, do_forward):
        if do_forward:
            lang_by_stage_fn = pp_lang_by_stages_v1
            step_idxs = range(self.num_steps)
        else:
            lang_by_stage_fn = pp_lang_by_stages_v1_rev
            step_idxs = range(self.num_steps - 1, -1, -1)

        langs_by_stage = []
        step_idx_to_num_stages_dict = {}
        for step_idx in step_idxs:
            obj_name, cont_name = self.get_obj_cont_name(step_idx=step_idx)
            langs = lang_by_stage_fn(obj_name, cont_name)
            langs_by_stage.extend(langs)
            step_idx_to_num_stages_dict[step_idx] = len(langs)
        step_idx_to_num_stages_map = [
            step_idx_to_num_stages_dict[step_idx]
            for step_idx in sorted(step_idx_to_num_stages_dict.keys())]
        return langs_by_stage, step_idx_to_num_stages_map


def save_imarr(imarr, name):
    im = Image.fromarray(imarr)
    im.save(name)


def append_obs_dicts(obs_dict1, obs_dict2):
    """
    Appends obs_dict2 to obs_dict1
    Produces a list for each key

    If one of the dicts is empty, return the other.
    """
    if len(obs_dict2) == 0:
        return obs_dict1
    elif len(obs_dict1) == 0:
        return obs_dict2

    assert set(obs_dict1.keys()) == set(obs_dict2.keys())
    for key in obs_dict1:
        if isinstance(obs_dict1[key], list):
            obs_dict1[key].append(obs_dict2[key])
        elif not isinstance(obs_dict2[key], list):
            obs_dict1[key] = [obs_dict1[key], obs_dict2[key]]
        else:
            raise NotImplementedError
    return obs_dict1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=int, default=0)
    args = parser.parse_args()
    env = FrankaEnv()
    obs = env.reset()
    for t in range(args.ts):
        xyz_a = np.random.normal(scale=0.0, size=(3,))
        gripper_a = np.random.uniform(low=-1.0, high=-1.0, size=(1,))
        a = np.concatenate([xyz_a, np.zeros((2, )), gripper_a])
        obs, _, _, _ = env.step(a)
        print("obs['ee_pos']", obs['ee_pos'])
        # save_imarr(obs["image"], f"20230206_{t}.png")
