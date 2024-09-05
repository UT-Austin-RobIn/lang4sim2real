import argparse
from collections import Counter
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deoxys.envs import init_env
from deoxys.utils.control_utils import reset_joints
from deoxys.utils.params import *
from deoxys.utils.data_collection_utils import (
    get_obj_xy_pos, get_timestamp, is_camera_feed_live, paint_pp_rewards,
    PyGameListener)
from rlkit.torch.networks.cnn import R3MWrapper
from rlkit.torch.policies import GaussianStandaloneCNNPolicy, MakeDeterministic
from rlkit.torch.pretrained_models.language_models import (
    LM_STR_TO_FN_CLASS_MAP)
import rlkit.util.pytorch_util as ptu
from rlkit.util.experiment_script_utils import (
    create_task_indices_from_task_int_str_list,
    get_embeddings_from_str_list)


class EvalRealRobotRolloutFactory:
    def __init__(self, args):
        self.ckpt = self.load_ckpt(args.ckpt, args.cnn_type)
        self.ckpt_fpath = args.ckpt
        self.max_path_len = args.max_path_len

        # Init env
        self.env_name = args.env
        self.state_mode = 0
        kwargs = {}
        if args.state_mode is not None:
            kwargs['state_mode'] = args.state_mode
            self.state_mode = args.state_mode
        if self.env_name == "frka_wirewrap":
            kwargs['pause_in_violation_box'] = False
        self.obj_id = args.obj_id
        self.target_obj_name = OBJECT_DETECTOR_CLASSES[self.obj_id]
        kwargs['target_obj_name'] = self.target_obj_name
        self.env = init_env(args.env, **kwargs)

        self.multistep_env = self.env.num_steps > 1

        self.num_tasks = args.num_tasks
        self.eval_task_indices = create_task_indices_from_task_int_str_list(
            args.eval_task_idxs, args.num_tasks)
        self.num_rollouts_per_task = args.num_rollouts_per_task
        self.task_emb_input_mode = args.task_emb_input_mode
        self.task_embedding = args.task_embedding
        if self.task_embedding == "lang":
            self.lang_prefix = args.lang_prefix + " " if args.lang_prefix else ""
            self.lang_task_embs = self.get_lang_embeddings()

        self.max_num_forced_resets = 2
        # ^ Once it hits 2 forced resets, stop the rollout.
        self.pgl = PyGameListener()
        # ^ Click on this to force robot back to reset immediately and count as failure.

    def load_ckpt(self, ckpt_fpath, cnn_type):
        if cnn_type in ["resnet18", "beta-vae"]:
            ckpt = torch.load(
                ckpt_fpath,
                map_location=lambda storage, loc: storage.cuda(0))
            # model.load_state_dict(ckpt['model_state_dict'])
            self.policy = ckpt['evaluation/policy']
        elif cnn_type in ["clip", "r3m", "r3m-ft"]:
            policy_params = {'max_log_std': 0, 'min_log_std': -6, 'obs_dim': None, 'std_architecture': 'values', 'freeze_policy_cnn': True, 'lang_emb_obs_is_tokens': False, 'gripper_policy_arch': 'ac_dim', 'gripper_loss_type': None, 'output_dist_learning_cnn_embs_in_aux': False, 'aug_transforms': ['pad_crop'], 'image_augmentation_padding': 4, 'image_size': (128, 128, 3), 'batchnorm': False, 'hidden_sizes': [1024, 512, 256], 'std': None, 'output_activation': lambda x: x, 'cnn_output_stage': '', 'use_spatial_softmax': False, 'state_obs_dim': 22, 'emb_obs_keys': ['lang_embedding'], 'aux_tasks': [], 'aux_obs_bounds': {}, 'aux_to_feed_fc': 'none', 'obs_key_to_dim_map': {'image': 49152, 'lang_embedding': 384, 'state': 22}, 'observation_keys': ['image', 'lang_embedding', 'state'], 'film_emb_dim_list': [], 'action_dim': 7}
            cnn_params = {'clip_checkpoint': '/home/robin/Projects/albert/ckpts/clip/lr=1e-05_wd=0.1_agg=True=model=ViT-B/32_batchsize=128_workers=8_date=2022-09-23-02-43-39/checkpoints/epoch_1.pt', 'freeze': True, 'image_shape': (128, 128, 3), 'tokenize_scheme': 'clip', 'image_augmentation_padding': 0, 'task_lang_list': [], 'added_fc_input_size': 384 + 22}
            device = torch.device("cuda")
            if cnn_type == "clip":
                from rlkit.torch.networks.cnn import ClipWrapper
                policy_params["added_fc_input_size"] = cnn_params.pop(
                    "added_fc_input_size")
                cnn = ClipWrapper(**cnn_params)
                policy_params['cnn_out_dim'] = cnn.visual_outdim
                policy = GaussianStandaloneCNNPolicy(cnn=cnn, **policy_params)
            elif cnn_type in ["r3m", "r3m-ft"]:
                from r3m import load_r3m
                if cnn_type == "r3m":
                    cnn = load_r3m("resnet18")
                    policy_params['cnn_out_dim'] = cnn.module.outdim
                elif cnn_type == "r3m-ft":
                    kwargs = dict(
                        strict_param_load=False,
                        adapter_kwargs=dict(
                            compress_ratio=1),
                    )
                    cnn = R3MWrapper(device=device, **kwargs)
                    cnn = cnn.r3m
                    state_dict = torch.load(ckpt_fpath, map_location="cuda:0")
                    cnn.module.load_state_dict(state_dict['evaluation/r3m'])
                    policy_params['cnn_out_dim'] = cnn.module.outdim
                cnn.eval()
                cnn_params = {'added_fc_input_size': 406}
                policy = GaussianStandaloneCNNPolicy(
                    cnn=cnn, **policy_params, **cnn_params)
            policy.gripper_idx = 5  # eval_env.env.gripper_idx
            self.policy = MakeDeterministic(policy)
            state_dict = torch.load(ckpt_fpath, map_location="cuda:0")
            self.policy.load_state_dict(
                state_dict['evaluation/policy'], strict=False)
            ckpt = state_dict
            policy.to(device)
        else:
            raise NotImplementedError
        self.obs_keys = ckpt['evaluation/observation_keys']
        self.non_emb_obs_keys = ["image", "state"]
        assert set(self.non_emb_obs_keys).issubset(self.obs_keys)
        self.emb_obs_keys = list(
            set(self.obs_keys) - set(self.non_emb_obs_keys))
        self.policy.eval()

        return ckpt

    def get_lang_embeddings(self):
        eval_task_str_list = self.env.get_task_lang_dict()["instructs"]
        eval_task_str_list = [
            f"{self.lang_prefix}{x}" for x in eval_task_str_list]
        print(eval_task_str_list)
        variant = dict(
            lang_emb_model_type="minilm",
            finetune_lang_enc=False,
            l2_unit_normalize_target_embs=False,
        )
        emb_model_class = LM_STR_TO_FN_CLASS_MAP[
            variant['lang_emb_model_type']]
        emb_model = emb_model_class(
            l2_unit_normalize=variant['l2_unit_normalize_target_embs'],
            gpu=0)
        eval_embeddings = get_embeddings_from_str_list(
            emb_model, eval_task_str_list, variant, gpu=0)
        return eval_embeddings

    def flatten_im(self, o):
        o['image'] = o['image'].transpose(2, 0, 1).flatten() / 255.0

    def film_obs_processor(self, obs):
        out_dict = {}
        self.flatten_im(obs)
        out_dict["non_emb_obs"] = np.concatenate(
            [np.squeeze(obs[key]) for key in self.non_emb_obs_keys])
        out_dict["emb"] = [
            obs[emb_obs_key] for emb_obs_key in self.emb_obs_keys]
        return out_dict

    def obs_processor(self, obs):
        self.flatten_im(obs)
        return np.concatenate([obs[key] for key in self.obs_keys])

    def add_task_emb_to_obs(self, o, task_idx):
        assert isinstance(o, dict)
        if self.task_embedding == "onehot":
            onehot_vec = np.zeros((self.num_tasks,))
            onehot_vec[task_idx] = 1.0
            assert len(self.emb_obs_keys) == 1
            o.update({self.emb_obs_keys[0]: onehot_vec})
        elif self.task_embedding == "lang":
            o.update({self.emb_obs_keys[0]: self.lang_task_embs[task_idx]})
        else:
            raise NotImplementedError
        return o

    def try_get_obj_xy_pos(self, obj_name):
        try:
            obj_pos_in_robot_coords = get_obj_xy_pos(self.env, obj_name)
        except:
            obj_pos_in_robot_coords = None
        return obj_pos_in_robot_coords

    def pre_rollout_check(self, task_idx):
        if self.multistep_env:
            # Assert rews_by_step or rev_rews_by_step are 0.
            if task_idx % 2 == 0:
                all_steps_rews = self.env.reward_by_step(info={})
            else:
                all_steps_rews = self.env.rev_reward_by_step(info={})
            all_steps_unaccomplished = (np.array(all_steps_rews) == 0.0).all()

            if not all_steps_unaccomplished:
                print(
                    "Some steps were already completed. all_steps_rews: "
                    f"{all_steps_rews}")
                return False

            # check objects
            if task_idx % 2 == 0:
                for step_idx, obj_name in enumerate(
                        self.env.step_kwargs["obj_names"]):
                    obj_xy_pos = get_obj_xy_pos(self.env, obj_name)
                    rev_step_idx = self.env.num_steps - step_idx - 1
                    drop_offset_xy = self.env.rev_step_kwargs[
                        'drop_pos_offsets'][rev_step_idx][:2]
                    obj_lo_init = self.env.init_placement_distr[obj_name][
                        "low"]
                    obj_hi_init = self.env.init_placement_distr[obj_name][
                        "high"]
                    # subtract out drop_offset_xy since it was added to
                    # the x ~ (lo, hi) while dropping the obj
                    if step_idx == 1:
                        eps = 0.03
                        # ^ container had really tight bounds, loosen it
                    else:
                        eps = 0.0
                    above_lo = (
                        (obj_lo_init - eps)
                        <= (obj_xy_pos - drop_offset_xy)).all()
                    below_hi = (
                        (obj_xy_pos - drop_offset_xy)
                        <= (obj_hi_init + eps)).all()
                    if not (above_lo and below_hi):
                        print(
                            f"{obj_name} {obj_xy_pos - drop_offset_xy} not in "
                            f"bounds ({obj_lo_init}, {obj_hi_init})")
                        # return False
        else:
            # check object locations
            if task_idx % 2 == 0:
                obj_xy_pos = self.try_get_obj_xy_pos(self.target_obj_name)
                while obj_xy_pos is None:
                    input(f"Cannot find {self.target_obj_name}."
                          " Move into view of camera then press enter.")
                    obj_xy_pos = self.try_get_obj_xy_pos(self.target_obj_name)
                # obj_xy_pos = get_obj_xy_pos(self.env, self.target_obj_name)
                obj_xy_pos += OBJ_TO_PICK_XYZ_OFFSET[self.target_obj_name][:2]
                is_valid_pos = self.env.check_obj_xy_valid_init_pos(obj_xy_pos)
                if self.env_name == "frka_wirewrap":
                    self.env.obj_xy_init_polygon.show_pt(
                        obj_xy_pos, fname="eval_obj_bounds.png")
                if not is_valid_pos:
                    print(f"{self.target_obj_name} {obj_xy_pos} not in bounds")
                    return False
        return self.env.is_gripper_open(print_state=True)

    def perform_rollout_on_task_idx(self, task_idx):
        rewards = []
        gripper_states = []
        o = self.env.reset()

        # Pre rollout check.
        while not self.pre_rollout_check(task_idx):
            input("Fix precondition then press enter")

        # clear all abort mouse click events
        self.pgl.clear_events()

        num_forced_resets_in_rollout = 0

        for t in range(self.max_path_len):
            # Listen for any abort mouse clicks
            # (when clicking on red popup square)
            abort = self.pgl.check_mouse_click()
            if abort:
                print("ABORTING Trajectory")
                if len(rewards) == 0:
                    rewards.append(0)
                break

            start_time = time.time()
            orig_o = dict(o)  # used for checking camera feed is live.
            o = self.add_task_emb_to_obs(o, task_idx)
            get_action_kwargs = {}
            if self.task_emb_input_mode in [
                    "film", "film_video_concat_lang",
                    "film_lang_concat_video"]:
                o_dict = self.film_obs_processor(o)
                o_for_agent = o_dict['non_emb_obs']
                get_action_kwargs.update(film_inputs=o_dict['emb'])
            elif self.task_emb_input_mode in ["concat_to_image_embs"]:
                o_for_agent = self.obs_processor(o)
                o_for_agent = ptu.from_numpy(o_for_agent)
            else:
                o_for_agent = self.obs_processor(o)
            a, stats_dict, aux_outputs = self.policy.get_action(
                o_for_agent, **get_action_kwargs)
            print(
                f"t={t}", a, "gripper", o['state'][self.env.gripper_state_idx])
            time_overhead = time.time() - start_time
            # print("time_overhead", time_overhead)
            next_o, r, d, info = self.env.step(a.copy(), time_overhead)

            # Check that the camera feed didn't freeze.
            camera_feed_is_live = is_camera_feed_live(orig_o, next_o)

            if not camera_feed_is_live:
                print(
                    "camera froze up, quitting collection of this trajectory")
                input("Replug in the camera. Then press enter.")
                break

            # Check the robot didn't reset too many times due to violating
            # workspace bounds
            num_forced_resets_in_rollout += int(info['forced_reset'])
            if num_forced_resets_in_rollout >= self.max_num_forced_resets:
                break

            gripper_states.append(next_o['state'][self.env.gripper_state_idx])
            if self.multistep_env or self.env_name == "frka_wirewrap":
                rewards.append(r)
            o = next_o

        # Paint rewards
        # -- Block adapted from data collection
        if self.multistep_env:
            # go back to reset position to take a photo
            last_gripper_opened = self.env.is_gripper_open()
            step_rew_list_before_reset = (
                info["rews_by_step"] if task_idx % 2 == 0
                else info["rev_rews_by_step"])
            print("lifting before reset")
            self.env.step(np.array([0, 0, 0.6, 0, 0, -1]))
            reset_joints(
                self.env.robot_interface,
                self.env.type_to_controller_cfg_map["joint"], self.env.logger,
                reset_joint_pos_idx=self.env.gripper_type)

            # Get reward based on if last obs had gripper opened.
            if task_idx % 2 == 0:
                step_rew_list = self.env.reward_by_step(info={})
                idx = self.env.step_idx
            else:
                step_rew_list = self.env.rev_reward_by_step(info={})
                idx = self.env.rev_step_idx
            step_rew_list = self.env.maybe_modify_step_rew_list(
                step_rew_list, idx, last_gripper_opened)
            print("rews_by_step", step_rew_list)

            if ((step_rew_list_before_reset != step_rew_list)
                    or (idx != (step_rew_list + [0.0]).index(0.0))):
                # There is disagreement between the last step_rew_list before
                # and after the reset. resolve with human input.
                while True:
                    inp = input(
                        f"gripper ended opened: {last_gripper_opened}. "
                        "which rew_list do we accept? "
                        "0 ([0.0, 0.0]), 1 ([1.0, 0.0]), 2 ([1.0, 1.0])")
                    if inp == '0':
                        step_rew_list = [0.0, 0.0]
                        break
                    elif inp == '1':
                        step_rew_list = [1.0, 0.0]
                        break  # keep the second step_rew_list
                    elif inp == '2':
                        step_rew_list = [1.0, 1.0]
                        break
            successful_traj = np.array(step_rew_list).all()
        elif self.env_name == "frka_wirewrap":
            successful_traj = bool(rewards[-1] > 0.0)
            print("successful_traj", successful_traj)
            self.env.reset()
        else:
            obj_pos_in_robot_coords = get_obj_xy_pos(
                self.env, self.target_obj_name)
            obj_placed_in_cont = self.env.obj_xy_placed_in_cont(
                obj_pos_in_robot_coords)
            obj_placed_out_cont = self.env.obj_xy_placed_out_cont(
                obj_pos_in_robot_coords)

            if task_idx % 2 == 0:
                successful_traj = obj_placed_in_cont
            else:
                successful_traj = obj_placed_out_cont

            gripper_states_arr = np.array(gripper_states)
            if successful_traj:
                rewards = paint_pp_rewards(
                    self.env, self.max_path_len, successful_traj,
                    gripper_states_arr)
            else:
                # Open gripper if gripper ends closed.
                if not self.env.gripper_open(gripper_states_arr[-1]):
                    print("opening gripper at end of traj")
                    self.env.step(np.array([0., 0., 0., 0., 0., -1.]))
                rewards = np.zeros((self.max_path_len,))
            # -- End block adapted from data collection

        rewards_dict = dict(
            last_reward=float(successful_traj),
            rewards_list=rewards,
        )

        if self.multistep_env:
            rewards_dict["final_step_rews"] = step_rew_list
            rewards_dict["num_steps_succeeded"] = (
                (step_rew_list + [0.0]).index(0.0))
            # count how many 1's from the beginning there are.
            # the first 0.0 invalidates all subsequent 1's.
        return rewards_dict

    def perform_rollouts(self):
        if self.multistep_env:
            self.perform_rollouts_multistep()
        else:
            self.perform_rollouts_singlestep()
        self.pgl.quit()

    def perform_rollouts_singlestep(self):
        self.success_by_task_idx = Counter()
        for task_idx in tqdm(self.eval_task_indices):
            for i in range(self.num_rollouts_per_task):
                input(
                    f"Task {task_idx}, traj {i}/{self.num_rollouts_per_task}. "
                    "Press ENTER to continue.")
                rewards_dict = self.perform_rollout_on_task_idx(task_idx)
                success = rewards_dict["last_reward"]
                assert success in {0.0, 1.0}
                self.success_by_task_idx[task_idx] += success
                print(
                    "self.success_by_task_idx[task_idx]",
                    f"{self.success_by_task_idx[task_idx]}/{i+1}")
                print(self.ckpt_fpath)
                # self.num_steps_succeeded_by_task_idx[task_idx]
                # += num_steps_completed
        self.success_by_task_idx = dict([
            (k, v / self.num_rollouts_per_task)
            for k, v in self.success_by_task_idx.items()])

        df_dict = {
            "success_rate": [
                self.success_by_task_idx[task_idx]
                for task_idx in self.eval_task_indices],
        }
        print("df_dict", df_dict)
        df = pd.DataFrame.from_dict(data=df_dict)
        print(df)
        df_path = f"{self.env_name}_{get_timestamp()}.csv"
        df.to_csv(df_path)
        print("df_path\n", df_path)
        overall_success_rate = np.mean(df['success_rate'])
        print("overall_success_rate", overall_success_rate)

    def perform_rollouts_multistep(self):
        self.success_by_task_idx = Counter()
        num_steps_completed_list_by_task_idx = {}
        for task_idx in tqdm(self.eval_task_indices):
            num_steps_completed_list_by_task_idx[task_idx] = []
            for i in range(self.num_rollouts_per_task):
                input(
                    f"Task {task_idx}, traj {i}/{self.num_rollouts_per_task}. "
                    "Press ENTER to continue.")
                rewards_dict = self.perform_rollout_on_task_idx(task_idx)
                success = rewards_dict["last_reward"]
                assert success in {0.0, 1.0}
                self.success_by_task_idx[task_idx] += success
                print(
                    "self.success_by_task_idx[task_idx]",
                    f"{self.success_by_task_idx[task_idx]}/{i+1}")
                print(self.ckpt_fpath)
                num_steps_completed_list_by_task_idx[task_idx].append(
                    rewards_dict["num_steps_succeeded"])
                print(
                    "num_steps_completed_list_by_task_idx",
                    num_steps_completed_list_by_task_idx[task_idx])
        self.success_by_task_idx = dict([
            (k, v / self.num_rollouts_per_task)
            for k, v in self.success_by_task_idx.items()])

        df_dict = {
            "success_rate": [
                self.success_by_task_idx[task_idx]
                for task_idx in self.eval_task_indices],
            "num_steps_completed": [
                np.mean(num_steps_completed_list_by_task_idx[task_idx])
                for task_idx in self.eval_task_indices],
            "num_steps_completed_list": [
                str(num_steps_completed_list_by_task_idx[task_idx])
                for task_idx in self.eval_task_indices],
        }
        print(
            "num_steps_completed_list_by_task_idx",
            num_steps_completed_list_by_task_idx)
        print("df_dict", df_dict)
        df = pd.DataFrame.from_dict(data=df_dict)
        print(df)
        df_path = f"{self.env_name}_{get_timestamp()}.csv"
        df.to_csv(df_path)
        print("df_path\n", df_path)
        overall_success_rate = np.mean(df['success_rate'])
        print("overall_success_rate", overall_success_rate)
        print("avg_num_steps_completed", np.mean(df['num_steps_completed']))


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # python deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_pp --state-mode 1 --task-embedding onehot --max-path-len 20 --num-tasks 2 --eval-task-idxs 0-1 --num-rollouts-per-task 10 --gpu 0
    # python deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_pp --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 20 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
    # python deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_obj_bowl_plate --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 50 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
    # python deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_wirewrap --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 50 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
    # python deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_wirewrap --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 50 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --cnn-type clip --gpu 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-len", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--eval-task-idxs",  nargs="+", type=str, default=[])
    parser.add_argument("--num-rollouts-per-task", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--state-mode", type=int, default=None, choices=[0, 1, None])
    parser.add_argument(
        "--task-embedding", type=str, default="onehot",
        choices=["onehot", "lang"])
    parser.add_argument(
        "--lang-prefix", type=str, required=True, default=None,
        choices=["", "Real:"])
    parser.add_argument("--cnn-type", type=str, default="resnet18", choices=[
        "resnet18", "r3m", "r3m-ft", "clip", "beta-vae"])
    parser.add_argument(
        "--task-emb-input-mode", default="film",
        choices=["film", "concat_to_image_embs"])
    args = parser.parse_args()

    if args.task_embedding == "lang":
        assert args.lang_prefix is not None

    if args.cnn_type in ["clip", "r3m", "r3m-ft", "beta-vae"]:
        args.task_emb_input_mode = "concat_to_image_embs"

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    eval_collector = EvalRealRobotRolloutFactory(args)
    eval_collector.perform_rollouts()
