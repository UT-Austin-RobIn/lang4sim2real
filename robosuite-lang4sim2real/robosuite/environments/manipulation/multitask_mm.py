import itertools

import cv2
import gym
import numpy as np

from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.utils.object_utils import OBJ_NAME_TO_XYZ_OFFSET
from robosuite.utils.observables import Observable, sensor


class MultitaskMM(PickPlace):
    def __init__(
            self, image_dim=None, image_hd_dim=None, transpose_image=False,
            state_mode=2, **kwargs):
        self.image_dim = image_dim
        self.image_hd_dim = image_hd_dim
        self.transpose_image = transpose_image
        assert state_mode in {0, 1, 2}
        self.state_mode = state_mode  # mode for concatenating state_keys
        self.task_id = 0
        self.vm_id = 0  # vm = verb-manner
        self.object_id = 0
        self.is_multitask_env = True
        self.is_mm_env = True

        # Multitask IDs
        self.set_vm_id_maps()
        self.reward_mode_to_vm_id_map = {
            v: k for k, v in self.vm_id_to_reward_mode_map.items()}
        self.set_obj_names()
        self.object_id_to_obj_str_map = dict(enumerate(self.obj_names))
        self.create_task_id_to_vm_obj_id_pair()
        self.num_tasks = len(self.task_id_to_vm_obj_id_pair_map)
        self.task_id_to_instruction_map = self.get_task_id_to_instruction_map()
        self.stack_z_margin = 0.04

        self.init_metrics_for_mm(obs=None)

        super().__init__(single_object_mode=4, **kwargs)

        # Standardize reward values across different tasks to be only in (0, 1)
        assert not self.reward_shaping
        assert self.reward_scale == 1.0
        self.mm_state_keys = ["ee_pos_speed_before_success"]

        if self.state_mode == 0:
            self.state_keys = [
                "robot0_eef_pos", "robot0_eef_quat",
                "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                "robot0_gripper_qpos", "robot0_gripper_qvel"]
        elif self.state_mode == 1:
            self.state_keys = [
                "robot0_eef_pos", "robot0_eef_quat",
                "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                "robot0_gripper_states"]
        elif self.state_mode == 2:
            self.state_keys = [
                "robot0_eef_pos", "robot0_eef_quat",
                "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                "robot0_gripper_states"] + self.mm_state_keys
        else:
            raise NotImplementedError

    def _set_observation_space(self):
        observation_space = {}
        if self.image_dim is not None:
            img_space = gym.spaces.Box(
                0, 1, (self.image_dim,), dtype=np.float32)
            observation_space['image'] = img_space
        observation_space['state'] = gym.spaces.Box(
            -1, 1, (self.state_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            observation_space)

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update)

        # for compatibility with railrl-ut
        # add image key
        image_keys = [obs_key for obs_key in obs.keys() if "image" in obs_key]
        if len(image_keys) == 1:
            image = cv2.resize(
                obs[image_keys[0]], (self.image_dim, self.image_dim))
            if self.transpose_image:
                image = np.transpose(image, (2, 0, 1))
            obs['image'] = np.array(image.flatten()) / 255.

            if self.image_hd_dim is not None:
                image_hd = cv2.resize(
                    obs[image_keys[0]], (self.image_hd_dim, self.image_hd_dim))
                if self.transpose_image:
                    image_hd = np.transpose(image_hd, (2, 0, 1))
                obs['image_hd'] = np.array(image_hd.flatten()) / 255.

        obs['state'] = np.concatenate([
            self._observables[key].obs
            if not self._observables[key]._is_number
            else np.array([self._observables[key].obs])
            for key in self.state_keys])

        return obs

    def set_vm_id_maps(self):
        self.vm_id_to_reward_mode_map = {
            0: ("stack", "quickly"),
            1: ("stack", "slowly"),
        }
        self.vm_id_to_horizon_map = {
            0: 100,
            1: 200,
        }

        # Only used in pick_place parent class
        self.verb_id_to_reward_mode_map = dict(
            [(_id, vm[0])
             for _id, vm in self.vm_id_to_reward_mode_map.items()])

    def set_obj_names(self):
        self.obj_names = ["Milk", "Bread", "Can", "Cereal small"]

    def get_verb_str(self, vm_id):
        return self.vm_id_to_reward_mode_map[vm_id][0]

    def get_mm_str(self, vm_id):
        """mm = manner of motion"""
        return self.vm_id_to_reward_mode_map[vm_id][1]

    def get_obj_str(self, obj_id):
        return self.object_id_to_obj_str_map[obj_id]

    def get_instruction_str(self, task_id):
        vm_id, obj_id = self.task_id_to_vm_obj_id_pair_map[task_id]
        obj_str = self.get_obj_str(obj_id)
        verb_str = self.get_verb_str(vm_id)
        mm_str = self.get_mm_str(vm_id)
        vm_id_to_template_instruction_map = {
            0: f"{mm_str} {verb_str} {obj_str} on flat block",
            1: f"{mm_str} {verb_str} {obj_str} on flat block",
        }
        return vm_id_to_template_instruction_map[vm_id]

    def get_task_id_to_instruction_map(self):
        task_id_to_instruction_map = dict()
        for task_id in self.task_id_to_vm_obj_id_pair_map:
            task_id_to_instruction_map[task_id] = self.get_instruction_str(
                task_id)
        return task_id_to_instruction_map

    def get_task_lang_dict(self):
        instructs = []
        for task_idx in range(self.num_tasks):
            task_instruct = self.get_instruction_str(task_idx)
            instructs.append(task_instruct)
        task_lang_dict = dict(
            instructs=instructs,
            # target_objs=target_objs,
        )
        return task_lang_dict

    def create_task_id_to_vm_obj_id_pair(self):
        all_vm_ids = sorted(list(self.vm_id_to_reward_mode_map.keys()))
        all_obj_ids = list(range(len(self.obj_names)))
        self.all_vm_obj_id_pairs = list(
            itertools.product(all_vm_ids, all_obj_ids))
        self.task_id_to_vm_obj_id_pair_map = {}
        for task_id, vm_obj_id_pair in enumerate(self.all_vm_obj_id_pairs):
            self.task_id_to_vm_obj_id_pair_map[task_id] = vm_obj_id_pair
        # Create reverse of list
        self.vm_obj_id_pair_to_task_id_map = dict(
            [(v, k) for k, v in self.task_id_to_vm_obj_id_pair_map.items()])

    def set_task_id(self, task_id):
        if isinstance(task_id, np.ndarray):
            task_id = int(task_id)
        # print("task_id", task_id)
        self.task_id = task_id

        # Set verb-motion and object ids
        vm_obj_id_pair = self.task_id_to_vm_obj_id_pair_map[task_id]
        self.vm_id, self.object_id = vm_obj_id_pair

    def set_goal(self, task_id):
        """Called by robomimic/envs/env_robosuite.py"""
        self.set_task_id(task_id)

    def get_horizon_from_task_id(self, task_id=None):
        if task_id is None:
            task_id = self.task_id
        vm_id, _ = self.task_id_to_vm_obj_id_pair_map[task_id]
        return self.vm_id_to_horizon_map[vm_id]

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        no_mm_metric_reset = kwargs.get("no_mm_metric_reset", False)
        if not no_mm_metric_reset:
            self.init_metrics_for_mm(obs)
        return obs

    def step(self, action):
        observations, reward, done, info = super().step(action)
        self.update_metrics_for_mm(observations)
        return observations, reward, done, info

    def init_metrics_for_mm(self, obs=None):
        self.total_ee_pos_dist = 0.
        self.steps_before_success_since_reset = 0.
        if obs:
            self.prev_ee_pos = obs["robot0_eef_pos"]
        self.seen_task_success = False

    def update_metrics_for_mm(self, obs):
        """Meant to be updated every step"""
        task_success_this_ts = bool(
            self.get_task_reward_by_task_id(self.task_id) and
            (self.steps_before_success_since_reset >= 1))
        # don't flip this when 0 steps have passed. This gets flipped on
        # randomly during the env.reset_to() call in env_robosuite.py in
        # robomimic for some random reason.
        self.seen_task_success = self.seen_task_success or task_success_this_ts

        if not self.seen_task_success:
            ee_pos_dist = np.linalg.norm(
                obs["robot0_eef_pos"] - self.prev_ee_pos)
            if ee_pos_dist <= 0.1:
                # most of the time we expect ee_pos_dist to be around 0.01 for the fast case.
                # sometimes during robomimic trajectory extraction, the ee_pos fluctuates a lot
                # on the 2nd and 3rd timestep, which messes up the average.
                # ignore all readings where fluctuations are impossibly high.
                self.total_ee_pos_dist += ee_pos_dist
            self.steps_before_success_since_reset += 1

        self.prev_ee_pos = obs["robot0_eef_pos"]

    def get_goal(self):
        return np.array(self.task_id)

    def get_reward_mode(self):
        return self.vm_id_to_reward_mode_map[self.vm_id]

    def reward(self, action=None):
        # action is not actually being used by this function or its calls
        rew = self.get_reward_by_task_id(self.task_id)
        return rew

    def get_task_reward_by_task_id(self, task_id):
        """
        Calculates whether the task has been done,
        irrespective of manner of motion.
        """
        def get_task_reward_by_task_id_and_mode(
                task_id, reward_mode, obj_xyz, cont_xyz, obj_str=None):
            target_obj_height = obj_xyz[2]
            if obj_str is not None:
                obj_z_offset = OBJ_NAME_TO_XYZ_OFFSET[obj_str.lower()][2]
            else:
                obj_z_offset = 0.0
            table_height = self.model.mujoco_arena.table_top_abs[2]

            object_id = [
                i for i in range(len(self.objects))
                if self.objects[i].name == obj_str][0]

            verb_mode = reward_mode[0]

            if verb_mode == "grasp":
                reward = float(
                    target_obj_height > (table_height + 0.08 + obj_z_offset))
            elif verb_mode == "pick_place":
                reward = float(super()._check_success(object_id=object_id))
            elif verb_mode == "stack":
                stack_xy_offset = np.linalg.norm(cont_xyz[:2] - obj_xyz[:2])
                platform_height = cont_xyz[2]
                reward = float(
                    (target_obj_height < (
                        platform_height + self.stack_z_margin + obj_z_offset))
                    and (stack_xy_offset < 0.05)
                    and not self.is_grasping_any_obj(
                        active_objs=[self.objects[object_id]])
                )
            else:
                raise NotImplementedError
            assert reward in {0, 1}
            return reward

        vm_id, object_id = self.task_id_to_vm_obj_id_pair_map[task_id]
        reward_mode = self.vm_id_to_reward_mode_map[vm_id]
        obj_str = self.objects[object_id].name
        target_obj_xyz = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
        cont_xyz = self.sim.data.body_xpos[
            self.obj_body_id[self.stack_cont_name]]
        reward = get_task_reward_by_task_id_and_mode(
            task_id, reward_mode, target_obj_xyz, cont_xyz, obj_str=obj_str)
        return reward

    def get_reward_by_task_id(self, task_id):
        task_reward = self.get_task_reward_by_task_id(task_id)
        vm_id, object_id = self.task_id_to_vm_obj_id_pair_map[task_id]
        mm_mode = self.get_mm_str(vm_id)
        ee_pos_speed = self._observables["ee_pos_speed_before_success"].obs
        if mm_mode == "quickly":
            mm_reward = ee_pos_speed >= 0.009
        elif mm_mode == "slowly":
            mm_reward = ee_pos_speed <= 0.0055
        else:
            raise NotImplementedError
        return float(task_reward and mm_reward)  # both are in {0, 1}

    def _check_success(self):
        """the "task" key is used for rollout success metrics."""
        success_dict = dict([
            (f"task_{reward_mode}", False)
            for reward_mode in self.reward_mode_to_vm_id_map])

        for task_id in self.task_id_to_vm_obj_id_pair_map:
            success = bool(self.get_reward_by_task_id(task_id))
            if task_id == self.task_id:
                success_dict["task"] = success
                success_dict[f"task_{task_id}"] = success
            else:
                success_dict[f"task_{task_id}"] = False
            vm_id, object_id = self.task_id_to_vm_obj_id_pair_map[task_id]
            reward_mode = self.vm_id_to_reward_mode_map[vm_id]
            success_dict[f"task_{reward_mode}"] = success_dict[
                f"task_{reward_mode}"] or success

        return success_dict

    def _setup_observables(self, include_task_id=True):
        def onehot(idx, num_possible):
            ret = np.zeros(num_possible)
            ret[idx] = 1.0
            return ret

        @sensor(modality="task_id")
        def task_id(obs_cache):
            return onehot(self.task_id, self.num_tasks)

        @sensor(modality="task_id")
        def vm_id(obs_cache):
            return onehot(self.vm_id, len(self.vm_id_to_reward_mode_map))

        @sensor(modality="task_id")
        def object_id(obs_cache):
            # return np.array(self.object_id)
            return onehot(self.object_id, len(self.object_id_to_obj_str_map))

        @sensor(modality="mm")
        def ee_pos_speed_before_success(obs_cache):
            if self.steps_before_success_since_reset:
                speed = (
                    self.total_ee_pos_dist
                    / self.steps_before_success_since_reset)
            else:
                speed = 0.0
            return np.array([speed])

        observables = super()._setup_observables()
        if include_task_id:
            observables['vm_id'] = Observable("vm_id", vm_id)
            observables['object_id'] = Observable("object_id", object_id)
        observables['task_id'] = Observable("task_id", task_id)
        observables['ee_pos_speed_before_success'] = Observable(
            "ee_pos_speed_before_success", ee_pos_speed_before_success)
        return observables
