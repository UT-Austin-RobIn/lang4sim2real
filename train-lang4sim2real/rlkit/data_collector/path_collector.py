from collections import deque, OrderedDict
from functools import partial

import numpy as np
import torch

from rlkit.data_collector.base import PathCollector
from rlkit.data_collector.rollout_functions import rollout
from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    pairwise_diffs_matrix)
from rlkit.util.eval_util import create_stats_ordered_dict
import rlkit.util.pytorch_util as ptu


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            task_embedding_type=None,

            # Args to get the final image distance metrics
            policy_cnn_dist_fn="",
            task_idx_to_final_img_obs_map={},
            policy_cnn_ckpt="",
            eval_embeddings_map={},
            image_dim=None,

            **kwargs
    ):
        """

        :param env:
        :param policy:
        :param max_num_epoch_paths_saved: Maximum number of paths to save per
        epoch for computing statistics.
        :param rollout_fn: Some function with signature
        ```
        def rollout_fn(
            env, policy, max_path_length, *args, **kwargs
        ) -> List[Path]:
        ```

        :param save_env_in_snapshot: If True, save the environment in the
        snapshot.
        :param kwargs: Unused kwargs are passed on to `rollout_fn`
        """
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._rollout_fn = partial(
            rollout_fn,
            output_cnn_embs_in_aux=bool(policy_cnn_dist_fn is not None),
            **kwargs)

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot
        self.task_embedding_type = task_embedding_type

        # attrs for doing final-img-dist on demos vs rollouts.
        # This only works for lang/one-hot embs. Not demo embs.
        self.policy_cnn_dist_fn = policy_cnn_dist_fn
        if self.policy_cnn_dist_fn != "":
            self.task_idx_to_final_img_obs_map = task_idx_to_final_img_obs_map
            self.policy_cnn_from_ckpt = None
            if policy_cnn_ckpt != "":
                self.policy_cnn_from_ckpt = torch.load(policy_cnn_ckpt)
                self.policy_cnn_from_ckpt_uses_film = False
                if isinstance(self.policy_cnn_from_ckpt, dict):
                    # We passed in a dict as policy_cnn_ckpt.
                    # Get the cnn from the policy.
                    self.policy_cnn_from_ckpt = (
                        self.policy_cnn_from_ckpt['trainer/policy'].cnn)
                    self.policy_cnn_from_ckpt_uses_film = True
        self.eval_embeddings_map = eval_embeddings_map  # Assumes this is fed to film.
        self.image_dim = image_dim

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            multi_task=False,
            task_index=0,
            log_obj_info=False,
            singletask_buffer=False,
    ):
        paths = []
        if log_obj_info:
            infos_list = []
        num_steps_collected = 0
        rollouts_collected = 0
        if multi_task:
            self._env.reset_task(task_index)
        # print("num_steps", num_steps)

        while num_steps_collected < num_steps:
            # print("num_steps_collected", num_steps_collected)
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            obs_processor_kwargs = {}
            if self.task_embedding_type == "mcil":
                # Alternate between using z_demo and z_lang
                # for conditioning the policy;
                # MCIL doesn't use both at the same time.
                obs_processor_kwargs["emb_obs_key_idx"] = (
                    rollouts_collected % 2)
            try:
                rollouts = self._rollout_fn(
                    self._env,
                    self._policy,
                    max_path_length=max_path_length_this_loop,
                    obs_processor_kwargs=obs_processor_kwargs,
                )
                if log_obj_info:
                    path, infos = rollouts
                else:
                    path = rollouts

                self._env.reset()

                path_len = len(path['actions'])
                if (
                        path_len != max_path_length
                        and not path['terminals'][-1]
                        and discard_incomplete_paths
                ):
                    break
                num_steps_collected += path_len
                paths.append(path)
                rollouts_collected += 1
                if log_obj_info:
                    infos_list.append(infos)
            except Exception as e:
                self._env.reset()
                print("Exception in rollout_fn", e)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        if log_obj_info:
            return paths, infos_list
        else:
            return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        if ((self.policy_cnn_dist_fn != "")
                and (len(self.get_epoch_paths()) > 0)):
            final_img_obs_dist_stats = self.get_final_img_obs_stats()
            stats.update(final_img_obs_dist_stats)
        return stats

    def get_final_img_obs_stats(self):
        def add_pairwise_dist_to_stats(
                img_embs1, img_embs2, key_prefix, stats):
            dists = pairwise_diffs_matrix(
                img_embs1, img_embs2, self.policy_cnn_dist_fn)
            stats[f'{key_prefix} Mean'] = ptu.get_numpy(torch.mean(dists))
            stats[f'{key_prefix} Std'] = ptu.get_numpy(torch.std(dists))

        stats = {}
        for eval_task_idx, demos_final_img_obs in (
                self.task_idx_to_final_img_obs_map.items()):
            key_prefix = (
                f'final_img_obs_dist rollout_demo task_{eval_task_idx}')

            # Process all img obs for feeding into CNN
            rollout_final_img_obs = np.concatenate(
                [path['observations'][-1]['image'].reshape(
                    1, 3, self.image_dim, self.image_dim)
                 for path in self.get_epoch_paths()], axis=0)
            rollout_final_img_obs = ptu.from_numpy(rollout_final_img_obs)
            demos_final_img_obs = ptu.from_numpy(
                demos_final_img_obs.transpose(0, 3, 1, 2) / 255.)

            # FiLM Inputs
            film_inputs = [
                ptu.from_numpy(self.eval_embeddings_map[eval_task_idx])]

            with torch.no_grad():
                curr_cnn_paths_final_img_obs_embs = torch.cat(
                    [ptu.from_numpy(path['final_obs_cnn_emb'])
                     for path in self.get_epoch_paths()], axis=0)
                curr_cnn_demos_final_img_obs_embs = (
                    self._policy._action_distribution_generator.cnn(
                        demos_final_img_obs,
                        film_inputs=film_inputs,
                        output_stage="last-and-conv_channels"))['last']
                add_pairwise_dist_to_stats(
                    curr_cnn_paths_final_img_obs_embs,
                    curr_cnn_demos_final_img_obs_embs,
                    f'{key_prefix}/curr_cnn',
                    stats)

                if self.policy_cnn_from_ckpt is not None:
                    policy_cnn_kwargs = {}
                    if self.policy_cnn_from_ckpt_uses_film:
                        policy_cnn_kwargs['film_inputs'] = film_inputs
                    init_cnn_paths_final_img_obs_embs = (
                        self.policy_cnn_from_ckpt(
                            rollout_final_img_obs,
                            output_stage="last-and-conv_channels",
                            **policy_cnn_kwargs)['last'])
                    init_cnn_demos_final_img_obs_embs = (
                        self.policy_cnn_from_ckpt(
                            demos_final_img_obs,
                            output_stage="last-and-conv_channels",
                            **policy_cnn_kwargs)['last'])
                    add_pairwise_dist_to_stats(
                        init_cnn_paths_final_img_obs_embs,
                        init_cnn_demos_final_img_obs_embs,
                        f'{key_prefix}/init_cnn Mean',
                        stats)
        return stats

    def get_snapshot(self):
        policy = self._policy._action_distribution_generator
        if hasattr(policy, "cnn") and hasattr(policy.cnn, "clip"):
            policy_state_dict = self._policy.state_dict()
            non_clip_policy_state_dict = dict(
                [(k, v) for k, v in policy_state_dict.items()
                 if "cnn.clip" not in k])
            snapshot_dict = dict(
                policy=non_clip_policy_state_dict,
                clip=policy.cnn.clip.module.state_dict(),
            )
        elif (hasattr(policy.cnn, "module")
                and hasattr(policy.cnn.module, "convnet")):
            # R3M
            policy_state_dict = self._policy.state_dict()
            non_r3m_policy_state_dict = dict(
                [(k, v) for k, v in policy_state_dict.items()
                 if "cnn.module" not in k])
            snapshot_dict = dict(
                policy=non_r3m_policy_state_dict,
                r3m=policy.cnn.module.state_dict(),
            )
        else:
            snapshot_dict = dict(policy=self._policy)
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_keys=['observation'],
            emb_obs_keys=[],
            task_embedding_type=None,
            task_emb_input_mode="concat_to_img_embs",
            emb_key_to_concat=None,
            aux_tasks=[],
            **kwargs
    ):
        # Create obs keys separated by emb and non_emb
        self.emb_obs_keys = list(emb_obs_keys)
        self.non_emb_obs_keys = []

        for obs_key in observation_keys:
            if obs_key not in self.emb_obs_keys:
                self.non_emb_obs_keys.append(obs_key)

        if task_emb_input_mode in [
                "film_video_concat_lang", "film_lang_concat_video"]:
            assert emb_key_to_concat in self.emb_obs_keys
            assert len(self.emb_obs_keys) == 2
            self.non_emb_obs_keys.append(emb_key_to_concat)  # should go at the end of the list
            self.emb_obs_keys.remove(emb_key_to_concat)

        def obs_processor(obs):
            return np.concatenate([obs[key] for key in observation_keys])

        def film_obs_processor(obs):
            out_dict = {}
            out_dict["non_emb_obs"] = np.concatenate(
                [np.squeeze(obs[key]) for key in self.non_emb_obs_keys])

            temp = []
            for emb_obs_key in self.emb_obs_keys:
                temp.append(obs[emb_obs_key])
            out_dict["emb"] = temp
            return out_dict

        def mcil_film_obs_processor(obs, emb_obs_key_idx):
            """
            emb_obs_key_idx is a number from
            {0, 1, ..., len(self.emb_obs_keys) - 1}
            that dictates which emb_obs_key (either lang or video)
            we pass to the policy as the task emb
            """
            assert emb_obs_key_idx in range(len(self.emb_obs_keys))
            out_dict = {}
            out_dict["non_emb_obs"] = np.concatenate(
                [np.squeeze(obs[key]) for key in self.non_emb_obs_keys])
            out_dict["emb"] = [obs[self.emb_obs_keys[emb_obs_key_idx]]]
            return out_dict

        if task_embedding_type == "mcil":
            assert len(self.emb_obs_keys) == 2
            preprocess_obs_for_policy_fn = mcil_film_obs_processor
        elif task_emb_input_mode in [
                "film", "film_video_concat_lang", "film_lang_concat_video"]:
            assert len(self.emb_obs_keys) > 0
            preprocess_obs_for_policy_fn = film_obs_processor
        else:
            preprocess_obs_for_policy_fn = obs_processor

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=preprocess_obs_for_policy_fn,
            task_emb_input_mode=task_emb_input_mode,
            aux_tasks=aux_tasks,
        )
        super().__init__(
            *args, rollout_fn=rollout_fn,
            task_embedding_type=task_embedding_type, **kwargs)
        self._observation_keys = observation_keys

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_keys=self._observation_keys,
        )
        return snapshot
