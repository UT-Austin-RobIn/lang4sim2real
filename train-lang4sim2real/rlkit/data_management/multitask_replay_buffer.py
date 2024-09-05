from collections import OrderedDict

from gym.spaces import Box, Discrete, Tuple
import numpy as np
import torch
import warnings

from rlkit.data_management.dataset_from_hdf5 import (
    MultitaskSequenceDataset, MultitaskDataLoader)
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.util.roboverse_utils import process_torch_image_arr_from_buffer
from rlkit.util.misc_functions import (
    get_rand_idxs_from_frame_ranges,
    seperate_gripper_action_from_actions,
)


class MultiTaskReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            task_indices,
            use_next_obs_in_context,
            sparse_rewards,
            use_ground_truth_context=False,
            ground_truth_tasks=None,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param task_indices: for multi-task setting
        """
        if env_info_sizes is None:
            env_info_sizes = {}
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = sparse_rewards
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.use_ground_truth_context = use_ground_truth_context
        self.task_indices = task_indices
        self.ground_truth_tasks = ground_truth_tasks
        if use_ground_truth_context:
            assert ground_truth_tasks is not None
        if sparse_rewards:
            env_info_sizes['sparse_reward'] = 1
        self.task_buffers = dict(
            [(idx, ReplayBuffer()) for idx in task_indices])
        self._max_replay_buffer_size = max_replay_buffer_size
        self._env_info_sizes = env_info_sizes

    def add_sample(self, task, observation, action, reward, terminal,
                   next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
            observation, action, reward, terminal,
            next_observation, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            try:
                batch = self.task_buffers[task].random_batch(batch_size)
            except KeyError:
                print(task)
        return batch

    def random_trajectory(
            self, task, batch_size, with_replacement=True, k=None,
            frame_ranges=[]):
        return self.task_buffers[task].random_trajectory(
            batch_size, with_replacement,
            k, frame_ranges)

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def add_multitask_paths(self, paths):
        for path in paths:
            task = path['env_infos'][0]['task_idx']
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def clear_all_buffers(self):
        for buffer in self.task_buffers.values():
            buffer.clear()

    def sample_batch(self, indices, batch_size):
        """
        sample batch of training data from a list of tasks for training the
        actor-critic.

        :param indices: task indices
        :param batch_size: batch size for each task index
        :return:
        """
        # this batch consists of transitions sampled randomly from buffer
        # rewards are always dense
        batches = [
            self.random_batch(idx, batch_size=batch_size) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        # unpacked = [torch.cat(x, dim=0) for x in unpacked]
        unpacked = [np.concatenate(x, axis=0) for x in unpacked]

        obs, actions, rewards, next_obs, terms, gripper_actions = unpacked
        return {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_obs,
            'terminals': terms,
            'gripper_actions': gripper_actions,
        }

    def sample_batch_of_trajectories(self, indices, batch_size):
        """
        sample batch of trajectories from a list of tasks

        :param indices: task indices
        :param batch_size: batch size for each task index
        :return:
            [num_tasks, batch_size, traj_len, dim] array
        """
        batches = [
            self.random_trajectory(idx, batch_size=batch_size)
            for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [np.concatenate(x, axis=0) for x in unpacked]

        obs, actions, rewards, next_obs, terms = unpacked
        return {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_obs,
            'terminals': terms,
        }

    def sample_bcz_video_batch_of_trajectories(
            self, indices, batch_size, with_replacement=True, k=None,
            frame_ranges=[], unflatten_im=True):
        # import time
        # st = time.time()
        batches = [
            self.random_trajectory(
                idx, batch_size, with_replacement, k, frame_ranges)
            for idx in indices]
        # print("batches time:", time.time() - st)
        # st = time.time()

        aggregated_visual_batch_dict = {}
        for batch in batches:
            visual_batch_dict = self.video_encoder.process_visual_batch(
                batch, unflatten_im)
            for key, visual_batch in visual_batch_dict.items():
                if key not in aggregated_visual_batch_dict:
                    aggregated_visual_batch_dict[key] = [visual_batch]
                else:
                    aggregated_visual_batch_dict[key].append(visual_batch)

        for key in aggregated_visual_batch_dict:
            aggregated_visual_batch_dict[key] = np.array(
                aggregated_visual_batch_dict[key])

        # print("video_inputs time:", time.time() - st)
        return aggregated_visual_batch_dict

    def load_clip_fns(
            self, clip_checkpoint, tokenize_scheme, gpu=0, freeze_clip=True):
        from heatmaps_clip import load_model_preprocess
        from clip import clip
        self.clip, self.preprocess = load_model_preprocess(
            clip_checkpoint, gpu=gpu, freeze_clip=freeze_clip)
        self.tokenize_fn = clip.get_tokenize_fn(tokenize_scheme)

    def sample_context(self, indices, batch_size):
        '''
        Sample batch of context from a list of tasks from the replay buffer
        '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [
            self.random_batch(
                idx,
                batch_size=batch_size,
                sequence=False)
            for idx in indices
        ]
        if any(b is None for b in batches):
            return None
        if self.use_ground_truth_context:
            return np.array([self.ground_truth_tasks[i] for i in indices])
        context = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        # context = [torch.cat(x, dim=0) for x in context]
        context = [np.concatenate(x, axis=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = np.concatenate(context[:-1], axis=2)
        else:
            context = np.concatenate(context[:-2], axis=2)
        return context

    def unpack_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if self.sparse_rewards:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]

        unpacked = [o, a, r, no, t]

        if self.sep_gripper_action:
            ag = batch['gripper_actions'][None, ...]
            unpacked.append(ag)

        return unpacked

    def get_snapshot(self):
        return dict()

    def get_diagnostics(self):
        return OrderedDict([
            ("task " + str(i) + " size", self.num_steps_can_sample(i))
            for i in self.task_indices])


class ObsDictMultiTaskReplayBuffer(MultiTaskReplayBuffer):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            task_indices,
            use_next_obs_in_context,
            sparse_rewards,
            observation_keys,
            internal_keys=None,
            path_len=None,
            use_ground_truth_context=False,
            ground_truth_tasks=None,
            env_info_sizes=None,
            video_encoder=None,
            sep_gripper_action=False,
            gripper_idx=None,
            gripper_smear_ds_actions=False,
    ):
        """
        :param max_replay_buffer_size (either int or dict[task_idx --> int]):
        :param env:
        :param task_indices: for multi-task setting
        """
        if env_info_sizes is None:
            env_info_sizes = {}

        self.max_replay_buffer_size = max_replay_buffer_size
        self.env = env
        self.observation_keys = observation_keys
        self.internal_keys = internal_keys

        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = sparse_rewards
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.use_ground_truth_context = use_ground_truth_context
        self.task_indices = task_indices
        self.ground_truth_tasks = ground_truth_tasks

        self.sep_gripper_action = sep_gripper_action
        if use_ground_truth_context:
            assert ground_truth_tasks is not None
        if sparse_rewards:
            env_info_sizes['sparse_reward'] = 1

        if isinstance(self.max_replay_buffer_size, dict):
            assert set(task_indices).issubset(
                self.max_replay_buffer_size.keys())
            task_idx_to_max_buf_size_map = self.max_replay_buffer_size
        elif isinstance(self.max_replay_buffer_size, int):
            task_idx_to_max_buf_size_map = dict(
                [(idx, self.max_replay_buffer_size) for idx in task_indices])
        else:
            raise TypeError

        self.task_buffers = dict([(idx, ObsDictReplayBuffer(
            task_idx_to_max_buf_size_map[idx],
            env,
            path_len=path_len,
            observation_keys=observation_keys,
            internal_keys=internal_keys,
            sep_gripper_action=self.sep_gripper_action,
            gripper_idx=gripper_idx,
            gripper_smear_ds_actions=gripper_smear_ds_actions,
            # env_info_sizes=env_info_sizes,
        )) for idx in task_indices])

        self._max_replay_buffer_size = max_replay_buffer_size
        self._env_info_sizes = env_info_sizes
        self.video_encoder = video_encoder


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs
        # installed if not running the rand_param envs
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size
        else:
            raise TypeError("Unknown space: {}".format(space))


class MultitaskReplayBufferHdf5(MultiTaskReplayBuffer):
    def __init__(
            self, hdf5_path, bsz, meta_bsz, video_encoder, task_emb, num_tasks,
            num_data_workers=0, hdf5_cache_mode="low_dim", subseq_len=1,
            demo_enc_frame_ranges=None, success_only=True,
            max_num_demos_per_task=None, task_to_max_num_demos_override_map={},
            task_idxs=[], obs_keys=[],
            task_idx_to_emb_map={}, sep_gripper_action=False,
            gripper_idx=6, gripper_smear_ds_actions=False,
            max_path_len=30, expand_action_dim_params={},
            # Phase 1-2 args
            dom2_img_dir="", policy_cnn_dist_fn="", train_val_split=False,
            phase1_2_debug_mode=0):
        self.bsz = bsz
        self.meta_bsz = meta_bsz
        self.video_encoder = video_encoder
        self.task_emb = task_emb
        self.hdf5_obs_keys = ["image", "state"]
        self.train_val_split = train_val_split

        if task_emb in ["onehot", "lang"]:
            # Include the emb_obs_key in self.obs_keys
            self.obs_keys = obs_keys
            assert len(self.obs_keys) == 3
            self.emb_obs_key = list(
                set(self.obs_keys) - set(self.hdf5_obs_keys))[0]
        else:
            self.emb_obs_key = None
            self.obs_keys = self.hdf5_obs_keys

        self.num_tasks = num_tasks
        self.task_idx_to_emb_map = task_idx_to_emb_map
        if self.task_emb == "lang":
            assert len(self.task_idx_to_emb_map) >= 1
            if len(self.task_idx_to_emb_map) == 1:
                warnings.warn("Only 1 lang emb found in entire buffer.")
        elif self.task_emb == "onehot":
            assert self.task_idx_to_emb_map is None
            self.task_idx_to_emb_map = self.create_onehot_task_idx_to_emb_map(
                task_idxs)

        self.task_idx_to_subseq_len_map = {}
        if isinstance(max_path_len, dict):
            print("subseq_len", subseq_len, "max_path_len", max_path_len)
            assert subseq_len == max(max_path_len.values())
            self.task_idx_to_subseq_len_map = dict(max_path_len)
        else:
            assert isinstance(max_path_len, int)
            self.task_idx_to_subseq_len_map = dict(
                [(task_idx, subseq_len) for task_idx in task_idxs])
        print("self.task_idx_to_subseq_len_map", self.task_idx_to_subseq_len_map)

        self.task_idx_to_num_subseq_per_batch = {}
        for task_idx in task_idxs:
            subseq_len = self.task_idx_to_subseq_len_map[task_idx]
            self.task_idx_to_num_subseq_per_batch[task_idx] = (
                self.calc_num_subseq_per_batch(subseq_len, bsz))
        print(
            "self.task_idx_to_num_subseq_per_batch",
            self.task_idx_to_num_subseq_per_batch)

        dataset_keys = ["actions"]
        self.img_lang_dist_learning = False
        if dom2_img_dir != "":
            self.img_lang_dist_learning = True
            dataset_keys.append("lang_idx")

        # transition individually to minimize I/O operations.
        self.cache_demos_for_enc = True

        seq_ds_kwargs = {}
        if self.train_val_split:
            seq_ds_kwargs['filter_by_attribute'] = "train"

        self.train_dataset = MultitaskSequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=self.hdf5_obs_keys,
            dataset_keys=dataset_keys,
            hdf5_cache_mode=hdf5_cache_mode,
            task_idx_to_seq_length_map=self.task_idx_to_subseq_len_map,
            pad_seq_length=False,
            cache_demos_for_enc=self.cache_demos_for_enc,
            demo_enc_frame_ranges=demo_enc_frame_ranges,
            success_only=success_only,
            max_num_demos_per_task=max_num_demos_per_task,
            task_to_max_num_demos_override_map=(
                task_to_max_num_demos_override_map),
            task_idxs=task_idxs,
            gripper_idx=gripper_idx,
            gripper_smear_ds_actions=gripper_smear_ds_actions,
            max_path_len=max_path_len,
            expand_action_dim_params=expand_action_dim_params,
            **seq_ds_kwargs,
        )
        self.train_dataloader = MultitaskDataLoader(
            self.train_dataset, self.task_idx_to_num_subseq_per_batch,
            num_data_workers=num_data_workers)

        if self.train_val_split:
            self.val_dataset = MultitaskSequenceDataset(
                hdf5_path=hdf5_path,
                obs_keys=self.hdf5_obs_keys,
                dataset_keys=dataset_keys,
                hdf5_cache_mode=hdf5_cache_mode,
                task_idx_to_seq_length_map=self.task_idx_to_subseq_len_map,
                pad_seq_length=False,
                cache_demos_for_enc=self.cache_demos_for_enc,
                demo_enc_frame_ranges=demo_enc_frame_ranges,
                filter_by_attribute="valid",
                success_only=success_only,
                max_num_demos_per_task=max_num_demos_per_task,
                task_to_max_num_demos_override_map=(
                    task_to_max_num_demos_override_map),
                task_idxs=task_idxs,
                gripper_idx=gripper_idx,
                gripper_smear_ds_actions=gripper_smear_ds_actions,
                max_path_len=max_path_len,
                expand_action_dim_params=expand_action_dim_params,
            )
            self.val_dataloader = MultitaskDataLoader(
                self.val_dataset, self.task_idx_to_num_subseq_per_batch,
                num_data_workers=num_data_workers)

        self.task_indices = self.train_dataset.task_idxs
        self.sep_gripper_action = sep_gripper_action
        self.gripper_idx = gripper_idx

        # Phase 1-2 Args
        if self.img_lang_dist_learning:
            self.phase1_2_debug_mode = phase1_2_debug_mode
            from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
                compute_xdomain_lang_embs_diff_mat,
                get_train_val_loader)
            (self.dom2_img_lang_train_loader, self.dom2_img_lang_val_loader,
                dom2_img_lang_ds) = get_train_val_loader(
                dom2_img_dir,
                bool(policy_cnn_dist_fn == "dotprod"),
                self.bsz * self.meta_bsz,
                drop_last=True)
            self.dom2_lang_idx_to_emb_mat = (
                dom2_img_lang_ds.unique_lang_idx_to_lang_emb_matrix)
            self.img_lang_lang_enc = dom2_img_lang_ds.lang_enc
            self.dom2_img_lang_train_loader_iter = iter(
                self.dom2_img_lang_train_loader)

            self.dom1_lang_idx_to_emb_mat = (
                self.get_lang_idx_to_emb_mat_from_hdf5())

            self.xdomain_lang_embs_diff_mat = (
                compute_xdomain_lang_embs_diff_mat(
                    self.dom1_lang_idx_to_emb_mat,
                    self.dom2_lang_idx_to_emb_mat, device="cuda:0"))

    def calc_num_subseq_per_batch(self, subseq_len, bsz):
        if subseq_len >= 10:
            min_num_subseq = bsz / subseq_len
            if min_num_subseq < 5:
                # Sample double the amount if too little.
                num_subseq_per_batch = int(np.ceil(2 * min_num_subseq))
            else:
                num_subseq_per_batch = int(np.ceil(min_num_subseq))
        elif subseq_len == 1:
            num_subseq_per_batch = bsz
        else:
            raise ValueError
        return num_subseq_per_batch

    def get_lang_idx_to_emb_mat_from_hdf5(self):
        # Involves self.train_dataset
        return self.img_lang_lang_enc(self.train_dataset.lang_desc_list)

    def get_xdomain_lang_embs_diff_mat(self):
        return self.xdomain_lang_embs_diff_mat

    def get_task_idx_to_final_img_obs_map(self, eval_tasks):
        task_idx_to_final_img_obs_map = {}
        for eval_task_idx in eval_tasks:
            if eval_task_idx in self.task_indices:
                demo_ids = self.train_dataset.task_idx_to_demo_id_list_map[
                    eval_task_idx]
                task_idx_to_final_img_obs_map[eval_task_idx] = []
                for demo_id in demo_ids:
                    last_img_obs = self.train_dataset.hdf5_file[
                        f"data/{eval_task_idx}/{demo_id}/observations/image"][-1]
                    task_idx_to_final_img_obs_map[eval_task_idx].append(
                        last_img_obs)
                task_idx_to_final_img_obs_map[eval_task_idx] = np.array(
                    task_idx_to_final_img_obs_map[eval_task_idx])
        return task_idx_to_final_img_obs_map

    def add_sample(self):
        raise NotImplementedError

    def random_batch(self, task, batch_size, validation=False):
        def flatten_traj_to_trans_batch_dict(batch_dict, idxs):
            new_batch_dict = dict()
            for key in batch_dict:
                if isinstance(batch_dict[key], dict):
                    new_batch_dict[key] = flatten_traj_to_trans_batch_dict(
                        batch_dict[key], idxs)
                elif torch.is_tensor(batch_dict[key]):
                    flattened_batch = torch.cat(list(batch_dict[key]))
                    new_batch_dict[key] = flattened_batch[idxs]
                else:
                    raise TypeError
            return new_batch_dict

        assert self.bsz == batch_size
        if validation:
            batch = self.val_dataloader.get_task_idx_batch(task)
        else:
            batch = self.train_dataloader.get_task_idx_batch(task)
        # Returns batch_size transitions
        idxs = np.random.randint(
            0,
            (self.task_idx_to_num_subseq_per_batch[task]
                * self.task_idx_to_subseq_len_map[task]),
            batch_size)
        batch = flatten_traj_to_trans_batch_dict(batch, idxs)
        batch['observations'] = self.maybe_add_emb_to_obs(
            batch['observations'], task)
        batch['observations'] = self.process_obs(batch['observations'])

        if self.sep_gripper_action:
            batch['actions'], batch['gripper_actions'] = (
                seperate_gripper_action_from_actions(
                    batch['actions'], self.gripper_idx))

        if self.img_lang_dist_learning:
            # Add language from dom1 to batch
            batch['lang_idx1'] = batch['lang_idx']

        return batch

    def sample_batch(self, indices, batch_size, validation=False):
        """
        sample batch of training data from a list of tasks for training the
        imitation learning agent.

        :param indices: task indices
        :param batch_size: batch size for each task index
        :return:
        """
        # this batch consists of transitions sampled randomly from buffer
        batches = [
            self.random_batch(
                idx, batch_size=batch_size, validation=validation)
            for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [np.concatenate(x, axis=0) for x in unpacked]

        if self.sep_gripper_action:
            obs, actions, gripper_actions = unpacked
        elif self.img_lang_dist_learning:
            obs, actions, dom1_lang_idx = unpacked
        else:
            obs, actions = unpacked

        batch = {}

        if self.img_lang_dist_learning:
            # Add dom2 obs and lang and dummy action
            try:
                dom2_obs, dom2_lang, dom2_lang_idx = next(
                    self.dom2_img_lang_train_loader_iter)
            except StopIteration:
                # reinit the iter
                self.dom2_img_lang_train_loader_iter = iter(
                    self.dom2_img_lang_train_loader)
                dom2_obs, dom2_lang, dom2_lang_idx = next(
                    self.dom2_img_lang_train_loader_iter)
            dom2_obs = dom2_obs.reshape(dom2_obs.shape[0], -1).cpu().numpy()
            dom2_obs = np.concatenate(
                [x[None] for x in np.split(dom2_obs, self.meta_bsz, axis=0)],
                axis=0)
            assert dom2_obs.shape[:2] == obs.shape[:2] == (
                self.meta_bsz, self.bsz)
            batch['lang_idx1'] = dom1_lang_idx
            batch['lang_idx2'] = np.concatenate(
                [x[None] for x in np.split(
                    dom2_lang_idx.cpu().numpy(), self.meta_bsz, axis=0)],
                axis=0)

            assert len(obs.shape) == len(dom2_obs.shape) == 3
            extra_dummy_dims = np.zeros((
                obs.shape[0], obs.shape[1], obs.shape[2] - dom2_obs.shape[2]))
            dom2_obs = np.concatenate([dom2_obs, extra_dummy_dims], axis=-1)
            assert obs.shape == dom2_obs.shape
            if self.phase1_2_debug_mode in [2, 3]:
                obs = np.concatenate([obs, obs], axis=0)
                actions = np.concatenate([actions, actions], axis=0)
            elif self.phase1_2_debug_mode in [4]:
                obs = np.concatenate([obs, obs], axis=0)
                actions = np.concatenate(
                    [actions, np.zeros(actions.shape)], axis=0)
            elif self.phase1_2_debug_mode in [5]:
                # replace dummy dims with those from obs
                obs_state = obs[:, :, -extra_dummy_dims.shape[-1]:]
                dom2_obs = np.concatenate(
                    [dom2_obs[:, :, :-obs_state.shape[-1]], obs_state],
                    axis=-1)
                obs = np.concatenate([obs, dom2_obs], axis=0)
                actions = np.concatenate([actions, actions], axis=0)
            elif self.phase1_2_debug_mode in [6]:
                obs_w_zero_state = np.concatenate(
                    [obs[:, :, :-extra_dummy_dims.shape[-1]], extra_dummy_dims],
                    axis=-1)
                obs = np.concatenate([obs, obs_w_zero_state], axis=0)
                actions = np.concatenate(
                    [actions, np.zeros(actions.shape)], axis=0)
            else:
                obs = np.concatenate([obs, dom2_obs], axis=0)
                actions = np.concatenate(
                    [actions, np.zeros(actions.shape)], axis=0)

        batch.update({
            'observations': obs,
            'actions': actions,
        })

        if self.sep_gripper_action:
            batch['gripper_actions'] = gripper_actions

        return batch

    def create_onehot_task_idx_to_emb_map(self, task_idxs):
        task_idx_to_emb_map = {}
        for task_idx in task_idxs:
            task_idx_to_emb_map[task_idx] = np.array([0] * self.num_tasks)
            task_idx_to_emb_map[task_idx][task_idx] = 1
        return task_idx_to_emb_map

    def maybe_add_emb_to_obs(self, obs_dict, task): 
        if self.emb_obs_key is not None:
            task_emb = torch.tensor(self.task_idx_to_emb_map[task])
            bsz = obs_dict['state'].shape[0]
            obs_dict[self.emb_obs_key] = torch.tile(task_emb, (bsz, 1))
        return obs_dict

    def process_obs(self, obs_dict):
        if "image" in obs_dict:
            obs_dict['image'] = process_torch_image_arr_from_buffer(
                obs_dict['image'].squeeze())

        obs_arr = []
        for obs_key in self.obs_keys:
            obs_arr.append(obs_dict[obs_key].squeeze())
        obs_arr = torch.cat(obs_arr, dim=1)
        return obs_arr

    def random_trajectory(
            self, task, batch_size, with_replacement=True, k=None,
            frame_ranges=[]):
        # with_replacement is not being looked at.
        assert batch_size <= self.bsz

        traj_im_obs_list = []
        for i in range(batch_size):
            if self.cache_demos_for_enc:
                traj_im_obs = (
                    self.train_dataset.get_task_idx_rand_demo_for_enc(task))
            else:
                traj = self.train_dataloader.get_task_idx_rand_traj(task)
                transition_rand_idxs = get_rand_idxs_from_frame_ranges(
                    frame_ranges, traj_start_idx=0)
                traj_im_obs = traj['observations']['image'][
                    transition_rand_idxs]
            traj_im_obs_list.append(traj_im_obs)

        # Returns a dict with single key:
        # {"observations": nparray with shape(batch_size, 48, 48, 3)}
        traj_dict = {"observations": np.array(traj_im_obs_list)}
        return traj_dict

    def sample_bcz_video_batch_of_trajectories(
            self, indices, batch_size, with_replacement=True, k=None,
            frame_ranges=[]):
        bcz_video_batch = super().sample_bcz_video_batch_of_trajectories(
            indices, batch_size, with_replacement=with_replacement, k=k,
            frame_ranges=frame_ranges, unflatten_im=False)
        return bcz_video_batch

    def num_steps_can_sample(self, task):
        return (len(self.train_dataloader.get_task_dataloader(task)) *
                self.task_idx_to_num_subseq_per_batch[task])

    def add_path(self, task, path):
        raise NotImplementedError

    def add_paths(self, task, paths):
        raise NotImplementedError

    def add_multitask_paths(self, paths):
        raise NotImplementedError

    def clear_buffer(self, task):
        raise NotImplementedError

    def clear_all_buffers(self):
        raise NotImplementedError

    def unpack_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        unpacked = [o, a]

        if self.sep_gripper_action:
            ag = batch['gripper_actions'][None, ...]
            unpacked.append(ag)
        elif self.img_lang_dist_learning:
            lang_idx = batch['lang_idx1'][None, ...]
            unpacked.append(lang_idx)

        return unpacked

    def get_snapshot(self):
        return OrderedDict([
            ("task " + str(i) + " size", self.num_steps_can_sample(i))
            for i in self.task_indices])


if __name__ == "__main__":
    # Example standalone use script.
    import time
    from rlkit.torch.networks.cnn import CNN
    from rlkit.util.video_encoders import BCZVideoEncoder

    hdf5_path = "/home/albert/dev/minibullet-ut/data/data/demo.hdf5"  # Change
    meta_bsz = 16
    bsz = 16
    frame_ranges = [(0, 1), (29, 30)]
    loss_kwargs = {}
    IMAGE_SIZE = (48, 48, 3)
    PLAIN_CNN_PARAMS = dict(
        input_width=IMAGE_SIZE[0],
        input_height=IMAGE_SIZE[1],
        input_channels=IMAGE_SIZE[2],
        kernel_sizes=[3, 3, 3],
        n_channels=[16, 16, 16],
        strides=[1, 1, 1],
        hidden_sizes=[1024, 512, 256],
        paddings=[1, 1, 1],
        pool_type='max2d',
        pool_sizes=[2, 2, 1],  # the one at the end means no pool
        pool_strides=[2, 2, 1],
        pool_paddings=[0, 0, 0],
        image_augmentation=True,
        image_augmentation_padding=4,
        output_size=8,
    )
    video_enc_net = CNN(**PLAIN_CNN_PARAMS)
    video_encoder = video_encoder = BCZVideoEncoder(
        video_enc_net,
        mosaic_rc=(1, 2),
        loss_type="cosine_dist",
        loss_kwargs=loss_kwargs,
        frame_ranges=frame_ranges,
        image_size=IMAGE_SIZE,
        use_cached_embs=False)
    rb = MultitaskReplayBufferHdf5(
        hdf5_path=hdf5_path, bsz=bsz,
        video_encoder=video_encoder, num_data_workers=0)

    for task_idx in rb.task_indices:
        s = time.time()
        rb.random_batch(task_idx, bsz)
        rb.random_trajectory(task_idx, bsz, frame_ranges=frame_ranges)
        print(f"time to load bsz={bsz}", time.time() - s)

    bcz_batch = rb.sample_bcz_video_batch_of_trajectories(
        rb.task_idxs, bsz, frame_ranges=frame_ranges)

    time.sleep(1)
    print("closing hdf5 file")
