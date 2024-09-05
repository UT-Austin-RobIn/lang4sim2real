import itertools
import math

import cv2
import gym
import numpy as np

from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import (
    StoveObject,
    PotWithHandlesObject,
    SpoolAndWireObject,
    PostObject,
    StringObject,
    StringObject2,
    PostObject2
)
from robosuite.utils.mjmod import DynamicsModder
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.object_utils import OBJ_NAME_TO_XYZ_OFFSET
from robosuite.wrappers.domain_randomization_wrapper import (
    DEFAULT_DYNAMICS_ARGS)


class Multitaskv1(PickPlace):
    def __init__(self, single_object_mode=3, **kwargs):
        assert single_object_mode in {3, 4}
        self.task_id = 0
        self.verb_id = 0
        self.object_id = 0
        self.is_multitask_env = True
        self.is_mm_env = False

        # Multitask IDs
        self.set_verb_id_maps()
        self.reward_mode_to_verb_id_map = {
            v: k for k, v in self.verb_id_to_reward_mode_map.items()}
        # self.set_object_to_id_map()
        # self.obj_id_to_obj_str_map = dict(
        #     [(v, k) for k, v in self.object_to_id.items()])
        self.set_obj_names()
        self.obj_id_to_obj_str_map = dict(enumerate(self.obj_names))
        self.create_task_id_to_verb_obj_id_pair()
        self.num_tasks = len(self.task_id_to_verb_obj_id_pair_map)
        self.set_verb_id_to_verb_str_map()
        self.task_id_to_instruction_map = self.get_task_id_to_instruction_map()
        self.stack_z_margin = 0.04
        # print(
        #     "self.task_id_to_instruction_map",
        #     self.task_id_to_instruction_map)

        super().__init__(single_object_mode=single_object_mode, **kwargs)

        # Standardize reward values across different tasks to be only in (0, 1)
        assert not self.reward_shaping
        assert self.reward_scale == 1.0

    def set_verb_id_maps(self):
        self.verb_id_to_reward_mode_map = {
            0: "grasp",
            1: "pick_place",
            2: "stack",
            3: "push",
        }
        self.verb_id_to_horizon_map = {
            0: 100,
            1: 200,
            2: 200,
            3: 100,
        }

    def set_obj_names(self):
        self.obj_names = ["Milk", "Bread", "Can"]  # "Cereal"

    def set_verb_id_to_verb_str_map(self):
        reward_mode_to_verb_str_map = {
            "pick_place": "put",
        }
        self.verb_id_to_verb_str_map = dict()
        for verb_id, verb_str in self.verb_id_to_reward_mode_map.items():
            if verb_str in reward_mode_to_verb_str_map:
                verb_str = reward_mode_to_verb_str_map[verb_str]
            self.verb_id_to_verb_str_map[verb_id] = verb_str

    def get_verb_str(self, verb_id):
        return self.verb_id_to_verb_str_map[verb_id]

    def get_obj_str(self, obj_id):
        return self.obj_id_to_obj_str_map[obj_id]

    def get_instruction_str(self, task_id):
        verb_id, obj_id = self.task_id_to_verb_obj_id_pair_map[task_id]
        obj_str = self.get_obj_str(obj_id)
        verb_str = self.get_verb_str(verb_id)
        verb_id_to_template_instruction_map = {
            0: f"{verb_str} {obj_str}",
            1: f"{verb_str} {obj_str} in bin",
            2: f"{verb_str} {obj_str} on flat block",
            3: f"{verb_str} {obj_str} to square zone",
        }
        return verb_id_to_template_instruction_map[verb_id]

    def get_task_id_to_instruction_map(self):
        task_id_to_instruction_map = dict()
        for task_id in self.task_id_to_verb_obj_id_pair_map:
            task_id_to_instruction_map[task_id] = self.get_instruction_str(
                task_id)
        return task_id_to_instruction_map

    def _create_obj_list_and_xy_ranges(self):
        # Used in _get_placement_initializer()
        if self.get_reward_mode() in ["grasp", "pick_place", "stack"]:
            return super()._create_obj_list_and_xy_ranges()
        elif self.get_reward_mode() == "push":
            bin_x_lohi, bin_y_lohi, y_margin = self._get_bin_xy_lohi()
            bin_x_lo, bin_x_hi = bin_x_lohi
            bin_x_mid = 0.5 * (bin_x_lo + bin_x_hi)
            bin_y_lo, bin_y_hi = bin_y_lohi
            bin_y_len = bin_y_hi - bin_y_lo

            distractor_objs = self.objects.copy()
            push_obj = [distractor_objs.pop(self.object_id)]

            push_zone = []
            for i, obj in enumerate(distractor_objs):
                if obj.name == "zone":
                    push_zone.append(distractor_objs.pop(i))

            push_side = np.random.choice(["left", "right"])
            if push_side == "left":
                push_y_range = [bin_y_lo, bin_y_lo + 0.25 * bin_y_len]
                distractor_obj_y_range = [
                    0.25 * bin_y_len, bin_y_hi + y_margin]
            elif push_side == "right":
                distractor_obj_y_range = [
                    bin_y_lo - y_margin, bin_y_lo + 0.75 * bin_y_len]
                push_y_range = [bin_y_lo + 0.75 * bin_y_len, bin_y_hi]
            else:
                raise NotImplementedError

            obj_list_and_xy_ranges = [
                (
                    "PushObjSampler",
                    push_obj,
                    [bin_x_lo, bin_x_mid],
                    push_y_range,
                    None,
                    0.0,
                ),
                (
                    "PushZoneSampler",
                    push_zone,
                    [bin_x_mid, bin_x_hi],
                    push_y_range,
                    None,
                    0.0,
                ),
                (
                    "CollisionObjectSampler",
                    distractor_objs,
                    [bin_x_lo, bin_x_hi],
                    distractor_obj_y_range,
                    None,
                    0.0,
                ),
            ]
            return obj_list_and_xy_ranges
        else:
            raise NotImplementedError

    def create_task_id_to_verb_obj_id_pair(self):
        all_verb_ids = sorted(list(self.verb_id_to_reward_mode_map.keys()))
        all_obj_ids = list(range(len(self.obj_names)))
        self.all_verb_obj_id_pairs = list(itertools.product(
            all_verb_ids, all_obj_ids))
        self.task_id_to_verb_obj_id_pair_map = {}
        for task_id, verb_obj_id_pair in enumerate(self.all_verb_obj_id_pairs):
            self.task_id_to_verb_obj_id_pair_map[task_id] = verb_obj_id_pair
        # Create reverse of list
        self.verb_obj_id_pair_to_task_id_map = dict(
            [(v, k) for k, v in self.task_id_to_verb_obj_id_pair_map.items()])

    def set_task_id(self, task_id):
        if isinstance(task_id, np.ndarray):
            task_id = int(task_id)
        # print("task_id", task_id)
        self.task_id = task_id

        # Set verb and object ids
        verb_obj_id_pair = self.task_id_to_verb_obj_id_pair_map[task_id]
        self.verb_id, self.object_id = verb_obj_id_pair

    def set_goal(self, task_id):
        """Called by robomimic/envs/env_robosuite.py"""
        self.set_task_id(task_id)

    def get_horizon_from_task_id(self, task_id=None):
        if task_id is None:
            task_id = self.task_id
        verb_id, _ = self.task_id_to_verb_obj_id_pair_map[task_id]
        return self.verb_id_to_horizon_map[verb_id]

    def get_goal(self):
        return np.array(self.task_id)

    def get_reward_mode(self):
        return self.verb_id_to_reward_mode_map[self.verb_id]

    def reward(self, action=None):
        # action is not actually being used by this function or its calls
        rew = self.get_reward_by_task_id(self.task_id)
        return rew

    def get_reward_by_task_id(self, task_id):
        verb_id, object_id = self.task_id_to_verb_obj_id_pair_map[task_id]
        reward_mode = self.verb_id_to_reward_mode_map[verb_id]
        obj_str = self.objects[object_id].name
        target_obj_xyz = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
        if reward_mode == "stack":
            cont_xyz = self.sim.data.body_xpos[
                self.obj_body_id[self.stack_cont_name]]
        elif reward_mode == "push":
            cont_xyz = self.sim.data.body_xpos[self.obj_body_id["zone"]]
        reward = self.get_reward_by_task_id_and_mode(
            task_id, reward_mode, target_obj_xyz, cont_xyz, obj_str=obj_str)
        return reward

    def get_reward_by_task_id_and_mode(
            self, task_id, reward_mode, obj_xyz, cont_xyz, obj_str=None):
        target_obj_height = obj_xyz[2]
        if obj_str is not None:
            obj_z_offset = OBJ_NAME_TO_XYZ_OFFSET[obj_str.lower()][2]
        else:
            obj_z_offset = 0.0
        table_height = self.model.mujoco_arena.table_top_abs[2]

        object_id = [
            i for i in range(len(self.objects))
            if self.objects[i].name == obj_str][0]

        if reward_mode == "grasp":
            reward = float(
                target_obj_height > (table_height + 0.08 + obj_z_offset))
        elif reward_mode == "pick_place":
            reward = float(super()._check_success(object_id=object_id))
        elif reward_mode == "stack":
            stack_xy_offset = np.linalg.norm(cont_xyz[:2] - obj_xyz[:2])
            platform_height = cont_xyz[2]
            reward = float(
                (target_obj_height < (
                    platform_height
                    + self.stack_z_margin
                    + obj_z_offset)) and
                (stack_xy_offset < 0.05) and
                not self.is_grasping_any_obj(
                    active_objs=[self.objects[object_id]])
            )
        elif reward_mode == "push":
            push_xy_offset = np.linalg.norm(cont_xyz[:2] - obj_xyz[:2])
            reward = float(
                (push_xy_offset < 0.04) and
                (target_obj_height < cont_xyz[2] + 0.04 + obj_z_offset) and
                not self.is_grasping_any_obj()
            )
        elif reward_mode == "wrap":
            reward = 0
        else:
            raise NotImplementedError

        assert reward in {0, 1}
        return reward

    def _check_success(self):
        """the "task" key is used for rollout success metrics."""
        success_dict = dict([
            (f"task_{reward_mode}", False)
            for reward_mode in self.reward_mode_to_verb_id_map])

        for task_id in self.task_id_to_verb_obj_id_pair_map:
            success = bool(self.get_reward_by_task_id(task_id))
            if task_id == self.task_id:
                success_dict["task"] = success
                success_dict[f"task_{task_id}"] = success
            else:
                success_dict[f"task_{task_id}"] = False
            verb_id, object_id = self.task_id_to_verb_obj_id_pair_map[task_id]
            reward_mode = self.verb_id_to_reward_mode_map[verb_id]
            success_dict[f"task_{reward_mode}"] = (
                success_dict[f"task_{reward_mode}"] or success)

        return success_dict

    def _setup_observables(self, include_task_id=True):
        def onehot(idx, num_possible):
            ret = np.zeros(num_possible)
            ret[idx] = 1.0
            return ret

        @sensor(modality="task_id")
        def task_id(obs_cache):
            # return np.array(self.task_id)
            return onehot(self.task_id, self.num_tasks)

        @sensor(modality="task_id")
        def verb_id(obs_cache):
            # return np.array(self.verb_id)
            return onehot(self.verb_id, len(self.verb_id_to_reward_mode_map))

        @sensor(modality="task_id")
        def object_id(obs_cache):
            # return np.array(self.object_id)
            return onehot(self.object_id, len(self.objects))

        observables = super()._setup_observables()
        # observables['task_id'] = Observable("task_id", task_id)
        if include_task_id:
            observables['verb_id'] = Observable("verb_id", verb_id)
            observables['object_id'] = Observable("object_id", object_id)
        return observables


class Multitaskv2(Multitaskv1):
    """Single object on scene but dictated by task_id"""
    def __init__(
            self,
            randomize="",
            image_dim=None,
            image_hd_dim=None,
            transpose_image=False,
            state_mode=0,
            **kwargs):
        super().__init__(single_object_mode=4, **kwargs)
        self.randomize = randomize
        if randomize == "wide":
            self.rbt_friction_range = [0, 8]
            self.rbt_damping_range = [0, 80]
        elif randomize == "narrow":
            self.rbt_friction_range = [0, 2]
            self.rbt_damping_range = [0, 20]
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

        self.image_dim = image_dim
        self.image_hd_dim = image_hd_dim
        self.transpose_image = transpose_image
        assert state_mode in {0, 1}
        self.state_mode = state_mode  # mode for concatenating state_keys
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
        else:
            raise NotImplementedError

        self.state_dim = sum([
            self._observables[state_key]._data_shape[0]
            for state_key in self.state_keys])
        self._set_observation_space()
        self._set_action_space()

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

    def _set_action_space(self):
        action_dim = 7
        act_high = np.ones(action_dim,)
        self.gripper_idx = action_dim - 1
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def set_verb_id_maps(self):
        self.verb_id_to_reward_mode_map = {
            0: "stack",
        }
        self.verb_id_to_horizon_map = {
            0: 200,
        }

    def get_instruction_str(self, task_id):
        verb_id, obj_id = self.task_id_to_verb_obj_id_pair_map[task_id]
        obj_str = self.get_obj_str(obj_id)
        verb_str = self.get_verb_str(verb_id)
        verb_id_to_template_instruction_map = {
            0: f"{verb_str} {obj_str} in bin",
        }
        return verb_id_to_template_instruction_map[verb_id]

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

    def reset_task(self, task_id):
        self.set_task_id(task_id)
        # for compatibility with railrl
        self.task_idx = self.task_id

    def set_obj_names(self):
        self.obj_names = ["Milk", "Bread", "Can", "Cereal"]

    def _setup_observables(self):
        def onehot(idx, num_possible):
            ret = np.zeros(num_possible)
            ret[idx] = 1.0
            return ret

        @sensor(modality="task_id")
        def task_id(obs_cache):
            # return np.array(self.task_id)
            return onehot(self.task_id, self.num_tasks)

        observables = super()._setup_observables(include_task_id=False)
        observables['task_id'] = Observable("task_id", task_id)
        return observables

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

    def reset(self):
        obs = super().reset()

        def set_rand_geom_color(indices, individual_materials=False):
            color = np.random.uniform(0, 1, size=(4,))
            color[3] = 1
            self.sim.model.geom_rgba[indices] = color
            if individual_materials:
                self.sim.model.geom_matid[indices] = np.random.randint(
                    0, self.sim.model.nmat - 1, size=(len(indices)))
            else:
                self.sim.model.geom_matid[indices] = np.random.randint(
                    0, self.sim.model.nmat - 1)

        if self.randomize != "":
            friction = (
                np.random.uniform(*self.rbt_friction_range)
                * self.orig_dof_frictionloss)
            damping = (
                np.random.uniform(*self.rbt_damping_range)
                * self.orig_dof_damping)
            for dof_idx in self.robots[0].joint_indexes:
                self.sim.model.dof_frictionloss[dof_idx] = friction[dof_idx]
                self.sim.model.dof_damping[dof_idx] = damping[dof_idx]
            for obj in self.objects:
                indices = [
                    self.sim.model.geom_name2id(g) + 1
                    for g in obj.contact_geoms]
                set_rand_geom_color(indices)
            rbt_indices = [
                self.sim.model.geom_name2id(g) + 1
                for g in self.robots[0].robot_model.contact_geoms]
            set_rand_geom_color(rbt_indices)
        return obs


class Multitaskv2_ang1(Multitaskv2):
    """Multitaskv2 with different camera angle"""
    def __init__(self, *args, **kwargs):
        return super().__init__(
            *args,
            camera_pos_objmode4=[-0.4643, -0.1997, 1.3394],
            camera_quat_objmode4=[0.6209, 0.3065, -0.3521, -0.6297],
            **kwargs)


class Multitaskv2_ang1_fr5damp50(Multitaskv2):
    """
    Multitaskv2 with different camera angle and higher friction
    and damping
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            camera_pos_objmode4=[-0.4643, -0.1997, 1.3394],
            camera_quat_objmode4=[0.6209, 0.3065, -0.3521, -0.6297],
            **kwargs)
        self.dynamics_modder = DynamicsModder(
            sim=self.sim,
            random_state=None,
            # meant to be a np.random.RandomState(seed) if seed specified.
            **DEFAULT_DYNAMICS_ARGS,
        )
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

    def reset(self):
        obs = super().reset()
        for dof_idx in self.robots[0].joint_indexes:
            self.sim.model.dof_frictionloss[dof_idx] = (
                5.0 * self.orig_dof_frictionloss[dof_idx])
            self.sim.model.dof_damping[dof_idx] = (
                50.0 * self.orig_dof_damping[dof_idx])
        return obs


class Multitaskv2_ang2(Multitaskv2):
    """
    Multitaskv2 with different camera angle.
    Used for images and figure creation.
    """
    def __init__(self, *args, **kwargs):
        return super().__init__(
            *args,
            camera_pos_objmode4=[
                0.8387995975695961, -0.84963113010538, 1.2812770278314753],
            camera_quat_objmode4=[
                0.6629404425621033, 0.5744012594223022,
                0.2989170551300049, 0.3757947087287903],
            **kwargs)


class PPObjToPotToStove(Multitaskv2):
    def __init__(self, randomize="", *args, **kwargs):
        assert randomize in ["wide", "narrow", ""]
        self.randomize = randomize
        if randomize == "wide":
            self.rbt_friction_range = [0, 8]
            self.rbt_damping_range = [0, 80]
        elif randomize == "narrow":
            self.rbt_friction_range = [0, 2]
            self.rbt_damping_range = [0, 20]

        self.step_kwargs = dict(
            reward_modes=["stack", "stack"],
            obj_names=["<target_obj>", "pot"],
            cont_names=["pot", "stove"],
        )
        self.step_kwargs["drop_pos_offsets"] = [
            np.array([-0.01, -0.03, OBJ_NAME_TO_XYZ_OFFSET[cont_name][2]])
            for cont_name in self.step_kwargs["cont_names"]]
        self.step_idx = 0
        super().__init__(*args, **kwargs)
        self.num_steps = len(self.step_kwargs["reward_modes"])
        self.stack_z_margin = 0.07
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

    def get_instruction_str(self, task_id):
        verb_id, obj_id = self.task_id_to_verb_obj_id_pair_map[task_id]
        full_instr = ""
        for step_idx in range(len(self.step_kwargs["reward_modes"])):
            verb_str = self.step_kwargs["reward_modes"][step_idx]
            obj_str = self.step_kwargs["obj_names"][step_idx]
            if obj_str == "<target_obj>":
                obj_str = self.get_obj_str(obj_id).lower()
            cont_str = self.step_kwargs["cont_names"][step_idx]
            verb_id_to_template_instruction_map = {
                0: f"{verb_str} {obj_str} on {cont_str}",
            }
            step_instr = verb_id_to_template_instruction_map[verb_id]
            if full_instr == "":
                prefix = ""
            else:
                prefix = ", then "
            full_instr += prefix + step_instr
        return full_instr

    def reset(self):
        self.step_idx = 0

        def set_rand_geom_color(indices, individual_materials=False):
            color = np.random.uniform(0, 1, size=(4,))
            color[3] = 1
            self.sim.model.geom_rgba[indices] = color
            if individual_materials:
                self.sim.model.geom_matid[indices] = np.random.randint(
                    0, self.sim.model.nmat - 1, size=(len(indices)))
            else:
                self.sim.model.geom_matid[indices] = np.random.randint(
                    0, self.sim.model.nmat - 1)

        obs = super().reset()
        if self.randomize != "":
            friction = (
                np.random.uniform(*self.rbt_friction_range)
                * self.orig_dof_frictionloss)
            damping = (
                np.random.uniform(*self.rbt_damping_range)
                * self.orig_dof_damping)
            for dof_idx in self.robots[0].joint_indexes:
                self.sim.model.dof_frictionloss[dof_idx] = friction[dof_idx]
                self.sim.model.dof_damping[dof_idx] = damping[dof_idx]
            for obj in self.objects:
                indices = [
                    self.sim.model.geom_name2id(g) + 1
                    for g in obj.contact_geoms]
                set_rand_geom_color(indices)
            rbt_indices = [
                self.sim.model.geom_name2id(g) + 1
                for g in self.robots[0].robot_model.contact_geoms]
            set_rand_geom_color(rbt_indices)
            stove_indices = [
                self.sim.model.geom_name2id(g) + 1
                for g in self.stove.contact_geoms]
            pot_indices = [
                self.sim.model.geom_name2id(g) + 1
                for g in self.pot.contact_geoms]
            set_rand_geom_color(stove_indices, True)
            set_rand_geom_color(pot_indices, True)
        return obs

    def set_obj_names(self):
        self.obj_names = ["Milk small", "Bread", "Can", "Cereal small"]

    def get_objs(self):
        objects = super().get_objs()
        objects = [obj for obj in objects if obj.name != "platform"]

        self.stove = StoveObject(name="stove")
        self.stack_cont_name = "pot"
        self.pot = PotWithHandlesObject(
            name=self.stack_cont_name,
            body_half_size=(0.06, 0.06, 0.045),
            handle_length=0.01,
            handle_width=0.01,
            thickness=0.005)
        objects.extend([self.stove, self.pot])
        # objects.extend([self.pot])

        return objects

    def _create_obj_list_and_xy_ranges(self):
        # Used in _get_placement_initializer()
        bin_x_lohi, bin_y_lohi, y_margin = self._get_bin_xy_lohi()
        bin_x_lo, bin_x_hi = bin_x_lohi
        bin_x_mid = 0.5 * (bin_x_lo + bin_x_hi)
        bin_x_len = bin_x_hi - bin_x_lo
        bin_y_lo, bin_y_hi = bin_y_lohi
        bin_y_len = bin_y_hi - bin_y_lo
        bin_y_mid = 0.5 * (bin_y_lo + bin_y_hi)

        stove_y_margin = 0.5 * y_margin
        pot_x_margin = 0.07

        distractor_objs = self.objects.copy()
        pp_obj = [distractor_objs.pop(self.object_id)]

        obj_pot_y_range = [bin_y_lo, bin_y_mid]
        stove_y_range = [bin_y_mid, bin_y_hi + stove_y_margin]

        obj_list_and_xy_ranges = [
            (
                "ObjSampler",
                pp_obj,
                [bin_x_mid, bin_x_hi],
                obj_pot_y_range,
                None,
                0.0,
            ),
            (
                "StoveSampler",
                self.stove,
                [bin_x_lo, bin_x_mid],
                stove_y_range,
                [0, 0],
                0.0,
            ),
            (
                "PotSampler",
                self.pot,
                [bin_x_lo - pot_x_margin, bin_x_mid],
                obj_pot_y_range,
                [0, 0],
                0.0,
            ),
        ]
        return obj_list_and_xy_ranges

    def reward(self, action=None, rews_by_step=None):
        # action is not actually being used by this function or its calls
        if rews_by_step is None:
            rews_by_step = self.reward_by_step(action)
        return rews_by_step[-1]

    def reward_by_step(self, action=None):
        rews_by_step = []
        for step_idx in range(self.num_steps):
            reward_mode = self.step_kwargs['reward_modes'][step_idx]

            obj_name = self.step_kwargs["obj_names"][step_idx]
            if obj_name == "<target_obj>":
                obj_name = self.objects[self.object_id].name
            obj_xyz = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
            cont_name = self.step_kwargs["cont_names"][step_idx]
            cont_xyz = self.sim.data.body_xpos[self.obj_body_id[cont_name]]

            step_rew = self.get_reward_by_task_id_and_mode(
                self.task_id, reward_mode, obj_xyz, cont_xyz, obj_str=obj_name)
            rews_by_step.append(step_rew)

        # Increment step_idx if succeeded current step
        if (rews_by_step[self.step_idx] == 1
                and self.step_idx < (self.num_steps - 1)):
            self.step_idx += 1
            self.stack_cont_name = (
                self.step_kwargs["cont_names"][self.step_idx])

        return rews_by_step

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        rews_by_step = self.reward_by_step(action)
        reward = self.reward(action, rews_by_step)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done

        # Store multistep info
        info = dict(
            rews_by_step=rews_by_step,
            step_idx=self.step_idx,
        )

        return reward, self.done, info


class PPObjToPotToStove_ang1_fr5damp50(PPObjToPotToStove):
    """
    PPObjToPotToStove with different camera angle and higher friction
    and damping
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            camera_pos_objmode4=[-0.4643, -0.1997, 1.3394],
            camera_quat_objmode4=[0.6209, 0.3065, -0.3521, -0.6297],
            **kwargs)
        self.dynamics_modder = DynamicsModder(
            sim=self.sim,
            random_state=None,
            # meant to be a np.random.RandomState(seed) if seed specified.
            **DEFAULT_DYNAMICS_ARGS,
        )
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

    def reset(self):
        obs = super().reset()
        for dof_idx in self.robots[0].joint_indexes:
            self.sim.model.dof_frictionloss[dof_idx] = (
                5.0 * self.orig_dof_frictionloss[dof_idx])
            self.sim.model.dof_damping[dof_idx] = (
                50.0 * self.orig_dof_damping[dof_idx])
        return obs


class WrapWire(Multitaskv2):
    def __init__(self, *args, **kwargs):
        self.step_kwargs = dict(
            reward_modes=["wrap"],
            obj_names=["spoolandwire"],
            cont_names=["spoolandwire"],
        )
        self.step_kwargs["drop_pos_offsets"] = [
            OBJ_NAME_TO_XYZ_OFFSET[cont_name][2]
            for cont_name in self.step_kwargs["cont_names"]]
        super().__init__(
            *args,
            use_table=True,
            table_friction=(0.1, 0.005, 0.0001),
            **kwargs)
        self.step_idx = 0
        self.num_steps = len(self.step_kwargs["reward_modes"])
        self.is_multitask_env = False

    def reset(self):
        obs = super().reset()
        self.step_idx = 0
        return obs

    def set_obj_names(self):
        self.obj_names = []

    def create_task_id_to_verb_obj_id_pair(self):
        self.verb_obj_id_pair_to_task_id_map = {(0, 0): 0}
        self.task_id_to_verb_obj_id_pair_map = {0: (0, 0), 1: (0, 0)}
        self.obj_id_to_obj_str_map = {0: "wire"}
        # self.id_to_object = {0: "wire"}
        # self.

    def set_object_to_id_map(self):
        self.object_to_id = {"wire": 0}

    def get_objs(self):
        objects = super().get_objs()
        objects = [obj for obj in objects if obj.name != "platform"]

        self.spool = SpoolAndWireObject(name="spoolandwire")
        objects.extend([self.spool])
        return objects

    def _create_obj_list_and_xy_ranges(self):
        # Used in _get_placement_initializer()
        bin_x_lohi, bin_y_lohi, y_margin = self._get_bin_xy_lohi()
        bin_x_lo, bin_x_hi = bin_x_lohi
        bin_x_mid = 0.5 * (bin_x_lo + bin_x_hi)
        bin_x_len = bin_x_hi - bin_x_lo
        bin_y_lo, bin_y_hi = bin_y_lohi
        bin_y_len = bin_y_hi - bin_y_lo
        bin_y_mid = 0.5 * (bin_y_lo + bin_y_hi)

        obj_list_and_xy_ranges = [
            (
                "SpoolSampler",
                self.spool,
                [bin_x_mid - .02, bin_x_mid],
                [bin_y_mid - .005, bin_y_mid + .025],
                [np.pi/2 - .1, np.pi/2 + .1],
                0.1,
            ),
        ]
        return obj_list_and_xy_ranges

    def reward(self, action=None, rews_by_step=None):
        obs = self._get_observations()
        spool_pos = obs["spoolandwire_geom_0_pos"]
        beads_poses = [obs[f"spoolandwire_geom_{i}_pos"] for i in range(1, 13)]

        beads_angles = [
            beads_pos[:2] - spool_pos[:2] for beads_pos in beads_poses]
        # ignore z
        beads_angles = [
            np.arctan2(*beads_angle) for beads_angle in beads_angles]
        total = 0
        for i in range(1, len(beads_angles)):
            angle = beads_angles[i]
            last_angle = beads_angles[i-1]
            if abs((2 * np.pi + angle) - last_angle) < abs(angle - last_angle):
                angle += 2*np.pi
            if (abs((-2 * np.pi + angle) - last_angle)
                    < abs(angle - last_angle)):
                angle -= 2*np.pi
            total += last_angle - angle
        if total >= 2 * np.pi:
            return [1]
        return [0]

    def get_reward_mode(self):
        return "wrap"


class WrapUnattachedWire(WrapWire):
    def __init__(self, initial_pos_both_sides, stages_mode, *args, **kwargs):
        self.initial_pos_both_sides = initial_pos_both_sides
        super().__init__(*args, **kwargs)
        self.success_wrap_ang = (5 / 3) * np.pi
        self.is_multitask_env = True
        self.num_tasks = 2
        self.stages_mode = stages_mode

    def get_task_id_to_instruction_map(self):
        task_id_to_instruction_str_map = {
            0: "wrap the beads counterclockwise around the cylinder",
            1: "wrap the beads clockwise around the cylinder",
            2: "unwrap the beads counterclockwise around the cylinder",
            3: "unwrap the beads clockwise around the cylinder",
        }
        return task_id_to_instruction_str_map

    def get_instruction_str(self, task_id):
        return self.get_task_id_to_instruction_map()[task_id]

    def get_objs(self):
        objects = super().get_objs()
        objects = [
            obj for obj in objects
            if obj.name != "platform" and obj.name != "spoolandwire"]

        self.spool = PostObject(name="spool")
        self.string = StringObject(name="spoolandwire")
        objects.extend([self.spool, self.string])
        return objects

    def _create_obj_list_and_xy_ranges(self):
        spool_x = -0.05
        spool_y = 0.25
        string_x = spool_x
        string_y = spool_y - 0.02

        distractor_objs = self.objects.copy()

        obj_list_and_xy_ranges = [
            (
                "SpoolSampler",
                self.spool,
                [spool_x - .02, spool_x],
                [spool_y - .005, spool_y + .025],
                [np.pi/2 - .1, np.pi/2 + .1],
                0.0,
            )]
        if self.initial_pos_both_sides:
            obj_list_and_xy_ranges.extend([
                (
                    "StringSampler1",
                    self.string,
                    [string_x - .2, string_x + .1],
                    [string_y - .1, string_y + .02],
                    [np.pi/2 - .1, np.pi/2 + .1],
                    0.1,
                ),
                (
                    "StringSampler2",
                    self.string,
                    [string_x - .2, string_x + .1],
                    [string_y + .29, string_y + .4],
                    [np.pi/2 - .1, np.pi/2 + .1],
                    0.1,
                ),
            ])
        else:
            obj_list_and_xy_ranges.extend([
                (
                    "BeadsSampler",
                    self.string,
                    [string_x - .2, string_x + .1],
                    [string_y - .1, string_y + .02],
                    [np.pi/2 - .1, np.pi/2 + .1],
                    0.1,
                )
            ])
        return obj_list_and_xy_ranges

    def get_angle_wrapped(self):
        obs = self._get_observations()
        spool_pos = obs["spool_geom_0_pos"]
        beads_poses = [obs[f"spoolandwire_geom_{i}_pos"] for i in range(0, 12)]

        beads_angles = [
            beads_pos[:2] - spool_pos[:2]
            for beads_pos in beads_poses]  # ignore z
        beads_angles = [
            np.arctan2(*beads_angle) for beads_angle in beads_angles]
        total = 0
        for i in range(1, len(beads_angles)):
            angle = beads_angles[i]
            last_angle = beads_angles[i-1]
            if abs((2 * np.pi + angle) - last_angle) < abs(angle - last_angle):
                angle += 2*np.pi
            if (abs((-2 * np.pi + angle) - last_angle)
                    < abs(angle - last_angle)):
                angle -= 2*np.pi
            total += last_angle - angle
        return total

    def reward(self, action=None, rews_by_step=None, threshold=None):
        unwrap = self.task_id == 2 or self.task_id == 3
        clockwise = self.task_id == 1 or self.task_id == 3
        total = self.get_angle_wrapped()
        if threshold is None:
            if unwrap:
                threshold = 0
            else:
                threshold = self.success_wrap_ang
                if clockwise:
                    threshold = self.success_wrap_ang * -1
        if (((not clockwise) and total >= threshold)
                or (clockwise and total <= threshold)):
            return 1
        return 0

    def get_reward_mode(self):
        return self.stages_mode

    def _check_success(self):
        """the "task" key is used for rollout success metrics."""
        success_dict = {}
        success_dict["task"] = bool(self.reward())
        success_dict["task_0"] = bool(self.reward())
        return success_dict

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward = self.reward()

        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        obs = self._get_observations()
        spool_pos = obs["spool_geom_0_pos"]
        beads_poses = [obs[f"spoolandwire_geom_{i}_pos"] for i in range(0, 12)]
        ee_dist_to_box = beads_poses[11] - spool_pos

        # Store multistep info
        info = dict(
            reward=reward,
            angle_wrapped=self.get_angle_wrapped(),
            ee_dist_to_box=ee_dist_to_box,
            is_pole_tipped_over=spool_pos[2] < 0.05,
            box_angle=math.atan2(ee_dist_to_box[0], ee_dist_to_box[1])
        )

        return reward, self.done, info


class WrapUnattachedWire_v2(WrapUnattachedWire):
    def __init__(self, randomize="", *args, **kwargs):
        assert randomize in ["wide", "narrow", ""]
        self.randomize = randomize
        if randomize == "wide":
            self.rbt_friction_range = [0, 8]
            self.rbt_damping_range = [0, 80]
        elif randomize == "narrow":
            self.rbt_friction_range = [0, 2]
            self.rbt_damping_range = [0, 20]
        
        super().__init__(
            initial_pos_both_sides=False,
            stages_mode="wrap-relative-location",
            *args,
            **kwargs)
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

    def reset(self):
        def set_rand_geom_color(indices):
            color = np.random.uniform(0, 1, size=(4,))
            color[3] = 1
            self.sim.model.geom_rgba[indices] = color

        obs = super().reset()
        if self.randomize != "":
            friction = (
                np.random.uniform(*self.rbt_friction_range)
                * self.orig_dof_frictionloss)
            damping = (
                np.random.uniform(*self.rbt_damping_range)
                * self.orig_dof_damping)
            for dof_idx in self.robots[0].joint_indexes:
                self.sim.model.dof_frictionloss[dof_idx] = friction[dof_idx]
                self.sim.model.dof_damping[dof_idx] = damping[dof_idx]
            for obj in self.objects:
                indices = [
                    self.sim.model.geom_name2id(g) + 1
                    for g in obj.contact_geoms]
                set_rand_geom_color(indices)
            rbt_indices = [
                self.sim.model.geom_name2id(g) + 1
                for g in self.robots[0].robot_model.contact_geoms]
            set_rand_geom_color(rbt_indices)
            table_indices = [
                self.sim.model.geom_name2id("table_collision") + 1]
            set_rand_geom_color(table_indices)
        return obs


class WrapUnattachedWire_ang1_fr5damp50(WrapUnattachedWire):
    """
    WrapUnattachedWire with different camera angle and higher friction
    and damping
    """
    def __init__(self, *args, **kwargs):
        kwargs["robots"] = ["PandaBlack"]
        super().__init__(
            *args,
            camera_pos_objmode4=[-0.51065, 0.047858, 1.24418],
            camera_quat_objmode4=[0.615395, 0.309568, -0.349420, -0.635108],
            **kwargs)
        self.dynamics_modder = DynamicsModder(
            sim=self.sim,
            random_state=None,
            # meant to be a np.random.RandomState(seed) if seed specified.
            **DEFAULT_DYNAMICS_ARGS,
        )
        self.orig_dof_frictionloss = np.array(self.sim.model.dof_frictionloss)
        self.orig_dof_damping = np.array(self.sim.model.dof_damping)

    def reset(self):
        obs = super().reset()
        for dof_idx in self.robots[0].joint_indexes:
            self.sim.model.dof_frictionloss[dof_idx] = (
                5.0 * self.orig_dof_frictionloss[dof_idx])
            self.sim.model.dof_damping[dof_idx] = (
                50.0 * self.orig_dof_damping[dof_idx])
        return obs

    def get_objs(self):
        objects = super().get_objs()
        objects = [
            obj for obj in objects
            if (obj.name != "platform"
                and obj.name != "spoolandwire"
                and obj.name != "spool")]

        self.spool = PostObject2(name="spool")
        self.string = StringObject2(name="spoolandwire")
        objects.extend([self.spool, self.string])
        return objects


class WrapUnattachedWire_ang1_fr5damp50_v2(WrapUnattachedWire_ang1_fr5damp50):
    def __init__(self, *args, **kwargs):
        super().__init__(
            initial_pos_both_sides=False,
            stages_mode="wrap-relative-location",
            *args,
            **kwargs)
