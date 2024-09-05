import argparse
from collections import Counter
from datetime import datetime
import os

import numpy as np
import pandas as pd
import roboverse
import torch
from tqdm import tqdm

from rlkit.torch.pretrained_models.language_models import (
    LM_STR_TO_FN_CLASS_MAP,
)
from rlkit.util.experiment_script_utils import (
    init_filled_buffer, load_buffer_datas, filter_buffer_if_needed,
    create_task_indices_from_task_int_str_list,
)
import rlkit.util.pytorch_util as ptu
from rlkit.util.roboverse_utils import get_buffer_size_multitask
from rlkit.util.visualization_utils import (
    load_task_idx_to_demos_map, get_z_embs
)


class EvalRolloutFactory:
    def __init__(self, args):
        self.args = args
        self.eval_video_batch_size = 1
        self.l2_unit_normalize_lang_embs = True
        if self.args.task_embedding == "lang":
            self.l2_unit_normalize_lang_embs = False
        self.eval_task_indices = create_task_indices_from_task_int_str_list(
            args.eval_task_idx_intervals, args.num_tasks)
        # print("self.eval_task_indices", self.eval_task_indices)

        self.env = self.load_env()
        self.set_obs_keys()
        self.task_idx_to_str_list_map = self.env.get_task_language_list()

        # flatten the list of lists into list of strs
        lst_of_lst_of_strs = self.task_idx_to_str_list_map
        task_idx_to_str_list_list = []
        for lst_of_strs in lst_of_lst_of_strs:
            if isinstance(lst_of_strs, list):
                assert len(lst_of_strs) == 1
                task_str = lst_of_strs[0]
            elif isinstance(lst_of_strs, str):
                task_str = lst_of_strs
            task_idx_to_str_list_list.append(task_str)
        self.task_idx_to_str_list_map = task_idx_to_str_list_list

        self.emb_model = self.load_emb_model()
        self.task_idx_to_lang_embs_map = self.get_lang_embs_from_str_list(
            self.task_idx_to_str_list_map)

    def load_ckpt(self, ckpt_fpath):
        ckpt = torch.load(
            ckpt_fpath,
            map_location=lambda storage, loc: storage.cuda(0))
        self.policy = ckpt['evaluation/policy']
        self.policy.eval()

        if self.args.task_embedding in ["demo", "demo_lang"]:
            self.demo_encoder = ckpt['trainer/task_encoder']
            self.demo_encoder.encoder.eval()

        if 'trainer/target_emb_net' in ckpt:
            self.target_emb_net = ckpt['trainer/target_emb_net']
            if self.target_emb_net is not None:
                raise NotImplementedError

        if self.args.task_embedding in ["demo", "demo_lang"]:
            self.task_idx_to_demo_embs_map = (
                self.get_demo_embs_from_target_buffer())

        return ckpt

    def load_env(self):
        kwargs = {
            "observation_img_dim": 48,
            "observation_img_hd_dim": 96,
            # "distractor_obj_hard_mode_prob": 1.0,
            # "deterministic_target_obj_referent": True,
        }
        env = roboverse.make(
            self.args.env, transpose_image=True, num_tasks=self.args.num_tasks,
            **kwargs)
        return env

    def set_obs_keys(self):
        self.observation_keys = ['image', 'state']

        if self.args.task_embedding == "demo_lang":
            self.emb_obs_keys = ["video_embedding", "lang_embedding"]
        elif self.args.task_embedding == "demo":
            self.emb_obs_keys = ["video_embedding"]
        elif self.args.task_embedding == "lang":
            self.emb_obs_keys = ["lang_embedding"]
        else:
            raise NotImplementedError

        self.non_emb_obs_keys = []
        for obs_key in self.observation_keys:
            if obs_key not in self.emb_obs_keys:
                self.non_emb_obs_keys.append(obs_key)

        if args.task_embedding == "demo_lang":
            if self.args.task_emb_input_mode == "film_video_concat_lang":
                emb_key_to_concat = "lang_embedding"
            elif self.args.task_emb_input_mode == "film_lang_concat_video":
                emb_key_to_concat = "video_embedding"

            if self.args.task_emb_input_mode in [
                    "film_video_concat_lang", "film_lang_concat_video"]:
                assert emb_key_to_concat in self.emb_obs_keys
                assert len(self.emb_obs_keys) == 2
                self.non_emb_obs_keys.append(emb_key_to_concat)
                # emb_key_to_concat should go at the end of the list
                self.emb_obs_keys.remove(emb_key_to_concat)

    def load_emb_model(self):
        emb_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
        emb_model = emb_model_class(
            l2_unit_normalize=self.l2_unit_normalize_lang_embs, gpu=0)
        return emb_model

    def get_lang_embs_from_str_list(self, task_strs):
        task_strs_tokenized = self.emb_model.tokenize_strs(task_strs)
        task_strs_tokenized = torch.Tensor(
            task_strs_tokenized).long().to("cuda:0")
        embeddings = self.emb_model(task_strs_tokenized)
        # embeddings is (300, 768)
        embeddings_map = {}
        for eval_task_idx in self.eval_task_indices:
            embeddings_map[eval_task_idx] = (
                embeddings[eval_task_idx].cpu().numpy())
        return embeddings_map

    def get_lang_emb_from_task_idx(self, task_idx):
        return self.task_idx_to_lang_embs_map[task_idx]

    def get_demo_embs_from_target_buffer(self):
        target_buffer_datas = load_buffer_datas(args.target_buffers)
        target_buffer_datas = filter_buffer_if_needed(
            target_buffer_datas, self.eval_task_indices)
        max_target_buffer_size = get_buffer_size_multitask(
            target_buffer_datas, success_only=True)
        variant = dict(
            task_embedding=self.args.task_embedding,
            max_path_length=self.args.max_path_len,
        )
        internal_keys = []
        target_buffer = init_filled_buffer(
            target_buffer_datas, variant, max_target_buffer_size,
            self.env, self.eval_task_indices, self.observation_keys,
            internal_keys, self.args.num_tasks, self.task_idx_to_lang_embs_map,
            success_only=True, video_encoder=self.demo_encoder)
        num_trajs_by_task_idx = dict([
            (i, target_buffer.task_buffers[i]._top // self.args.max_path_len)
            for i in self.eval_task_indices])
        min_num_trajs_per_task = min(num_trajs_by_task_idx.values())
        assert self.eval_video_batch_size <= min_num_trajs_per_task
        print("min_num_trajs_per_task", min_num_trajs_per_task)
        kwargs = dict(
            frame_ranges=self.demo_encoder.frame_ranges)

        # Calculate embs for each demo in target_buffer
        task_idx_to_demos_map = load_task_idx_to_demos_map(
            target_buffer,
            self.eval_task_indices,
            min_num_trajs_per_task,
            self.demo_encoder.k,
            kwargs, self.demo_encoder.process_visual_batch)

        task_idx_to_demo_embs_map = get_z_embs(
            task_idx_to_demos_map, self.task_idx_to_str_list_map,
            self.demo_encoder, self.env, modality_key="video")
        return task_idx_to_demo_embs_map

    def get_demo_emb_from_task_idx(self, task_idx):
        all_task_idx_demo_embs = self.task_idx_to_demo_embs_map[task_idx]
        demo_indices = np.random.choice(
            range(all_task_idx_demo_embs.shape[0]),
            size=self.eval_video_batch_size, replace=False)
        return all_task_idx_demo_embs[demo_indices]

    def get_task_emb_from_task_idx(self, task_idx):
        task_emb_dict = {}
        if self.args.task_embedding == "lang":
            task_emb_dict['lang_embedding'] = self.get_lang_emb_from_task_idx(
                task_idx)
        elif self.args.task_embedding == "demo":
            task_emb_dict['video_embedding'] = self.get_demo_emb_from_task_idx(
                task_idx)
        elif self.args.task_embedding == "demo_lang":
            task_emb_dict['lang_embedding'] = self.get_lang_emb_from_task_idx(
                task_idx)
            task_emb_dict['video_embedding'] = self.get_demo_emb_from_task_idx(
                task_idx)
        else:
            raise NotImplementedError
        return task_emb_dict

    def film_obs_processor(self, obs):
        out_dict = {}
        out_dict["non_emb_obs"] = np.concatenate(
            [np.squeeze(obs[key]) for key in self.non_emb_obs_keys])
        out_dict["emb"] = [
            obs[emb_obs_key] for emb_obs_key in self.emb_obs_keys]
        return out_dict

    def obs_processor(self, obs):
        return np.concatenate([obs[key] for key in self.observation_keys])

    def add_task_emb_to_obs(self, o, task_idx):
        assert isinstance(o, dict)
        o.update(self.get_task_emb_from_task_idx(task_idx))
        return o

    def perform_rollout_on_task_idx(self, task_idx):
        rewards = []
        o = self.env.reset()
        for t in range(self.args.max_path_len):
            o = self.add_task_emb_to_obs(o, task_idx)
            get_action_kwargs = {}
            if self.args.task_emb_input_mode in [
                    "film", "film_video_concat_lang",
                    "film_lang_concat_video"]:
                o_dict = self.film_obs_processor(o)
                o_for_agent = o_dict['non_emb_obs']
                get_action_kwargs.update(film_inputs=o_dict['emb'])
            else:
                o_for_agent = self.obs_processor(o)
            a, stats_dict, aux_outputs = self.policy.get_action(
                o_for_agent, **get_action_kwargs)

            next_o, r, d, env_info = self.env.step(a.copy())

            env_info['reward'] = r
            rewards.append(r)

            if isinstance(stats_dict, dict):
                env_info.update(stats_dict)

            o = next_o
        last_reward = rewards[-1]
        return last_reward

    def get_timestamp_str(self):
        x = datetime.now()
        timestamp_str = (
            f"{x.year}-{x.month}-{x.day}_{x.hour}-{x.minute}-{x.second}")
        return timestamp_str

    def perform_rollouts(self):
        self.success_by_task_idx = Counter()
        for task_idx in tqdm(self.eval_task_indices):
            for i in range(self.args.num_rollouts_per_task):
                success = self.perform_rollout_on_task_idx(task_idx)
                assert success in {0.0, 1.0}
                self.success_by_task_idx[task_idx] += success
        self.success_by_task_idx = dict([
            (k, [v/self.args.num_rollouts_per_task])
            for k, v in self.success_by_task_idx.items()])
        df = pd.DataFrame.from_dict(
            data=self.success_by_task_idx,
            orient='index', columns=["success_rate"])
        print(df)
        df_path = (
            f"{self.args.task_embedding}_{self.args.env}"
            f"_{self.get_timestamp_str()}.csv")
        df.to_csv(df_path)
        print("df_path\n", df_path)
        overall_success_rate = np.mean(df['success_rate'])
        print("overall_success_rate", overall_success_rate)


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument(
        "--ckpt", nargs="+", required=True, type=str,
        help="allows passing in multiple ckpts")
    parser.add_argument(
        "--target-buffers", type=str, nargs="+", default=[],
        help="optional buffer containing tasks we want to eval on")
    parser.add_argument("--task-embedding", type=str, choices=[
        'lang', 'onehot', 'none', 'demo', 'demo_lang', 'mcil'])
    parser.add_argument(
        "--task-emb-input-mode", type=str, required=True,
        choices=[
            "concat_to_img_embs", "film", "film_video_concat_lang",
            "film_lang_concat_video"])
    parser.add_argument(
        "--eval-task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument("--num-rollouts-per-task", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--max-path-len", type=int, default=30)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    if isinstance(args.ckpt, list):
        all_ckpts_to_eval = list(args.ckpt)

    rollout_factory = EvalRolloutFactory(args)

    for ckpt in all_ckpts_to_eval:
        rollout_factory.load_ckpt(ckpt)
        rollout_factory.perform_rollouts()
