from gym.spaces import Dict, Discrete
import numpy as np

import rlkit.data_management.images as image_np
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.util.misc_functions import (
    get_rand_idxs_from_frame_ranges,
    seperate_gripper_action_from_actions,
    smear_traj_gripper_actions,
)


class ObsDictReplayBuffer(ReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            path_len=None,
            ob_keys_to_save=None,
            internal_keys=None,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            save_data_in_snapshot=False,
            reward_dim=1,
            preallocate_arrays=False,

            sep_gripper_action=False,
            gripper_idx=None,
            gripper_smear_ds_actions=False,

            bias_point=None,
            before_bias_point_probability=0.5,
    ):
        """
        :param max_size:
        :param env:
        :param ob_keys_to_save: List of keys to save
        """
        if observation_key is not None and observation_keys is not None:
            raise ValueError(
                'Only specify observation_key or observation_keys')
        if observation_key is None and observation_keys is None:
            raise ValueError(
                'Specify either observation_key or observation_keys'
            )
        if observation_keys is None:
            observation_keys = [observation_key]
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.env = env
        self.path_len = path_len
        self.observation_keys = observation_keys
        self.save_data_in_snapshot = save_data_in_snapshot

        self.sep_gripper_action = sep_gripper_action
        self.gripper_idx = gripper_idx
        self.gripper_smear_ds_actions = gripper_smear_ds_actions

        # Args for biased sampling from the replay buffer
        self.bias_point = bias_point
        self.before_bias_point_probability = before_bias_point_probability

        self._action_dim = env.action_space.low.size
        self._actions = np.ones(
            (max_size, *env.action_space.shape),
            dtype=env.action_space.dtype,
        )
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.ones((max_size, 1), dtype='uint8')
        self.vectorized = reward_dim > 1
        self._rewards = np.ones((max_size, reward_dim))
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in observation_keys:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)
        for key in ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            module = image_np if key.startswith('image') else np
            self.arr_initializer = (
                module.ones if preallocate_arrays else module.zeros)
            self._obs[key] = self.arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=self.ob_spaces[key].dtype,
            )
            self._next_obs[key] = self.arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=self.ob_spaces[key].dtype,
            )

        self.ob_keys_to_save = ob_keys_to_save
        self._top = 0
        self._size = 0

        self._idx_to_future_obs_idx = np.ones((max_size, 2), dtype=np.int)

        if isinstance(self.env.action_space, Discrete):
            raise NotImplementedError

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        # raise NotImplementedError("Only use add_path")
        self._actions[self._top] = action
        self._terminals[self._top] = terminal
        self._rewards[self._top] = reward

        for key in self.ob_keys_to_save + self.internal_keys:
            self._obs[key][self._top] = observation[key]
            self._next_obs[key][self._top] = next_observation[key]

        self._top = (self._top + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path, ob_dicts_already_combined=False):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        # Maybe process actions
        if self.gripper_smear_ds_actions:
            actions = np.array([action for action in actions])
            actions = smear_traj_gripper_actions(
                actions, self.gripper_idx, (-1, 0, 1), path_len)
            actions = list(actions)

        if not ob_dicts_already_combined:
            obs = combine_dicts(obs, self.ob_keys_to_save + self.internal_keys)
            next_obs = combine_dicts(
                next_obs, self.ob_keys_to_save + self.internal_keys)

        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = (
                np.s_[self._top:self._top + num_pre_wrap_steps, ...]
            )
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, ...]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._rewards[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][
                        path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = [i, num_post_wrap_steps]
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = [i, num_post_wrap_steps]
        else:
            slc = np.s_[self._top:self._top + path_len, ...]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            self._rewards[slc] = rewards

            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = [i, self._top + path_len]
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size, biased_sampling):
        if biased_sampling:
            indices_any = np.random.randint(0, self._size, batch_size//2)
            indices_positive = np.argwhere(self._rewards)[:, 0]
            if indices_positive.shape[0] > 0:
                indices_positive = np.random.choice(
                    indices_positive, size=batch_size//2)
            else:
                indices_positive = np.random.randint(0, self._size,
                                                     batch_size//2)
            indices = np.concatenate([indices_any, indices_positive])
        else:
            if self.bias_point is not None:
                indices_1 = np.random.choice(
                    np.arange(self.bias_point), batch_size)
                indices_2 = np.random.choice(
                    np.arange(self.bias_point, self._size), batch_size)
                biased_coin_flip = (np.random.uniform(size=batch_size) <
                                    self.before_bias_point_probability)
                indices = np.where(biased_coin_flip, indices_1, indices_2)
            else:
                indices = np.random.randint(0, self._size, batch_size)
        return indices

    def random_batch(self, batch_size, biased_sampling=False):
        indices = self._sample_indices(batch_size, biased_sampling)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        if len(self.observation_keys) == 1:
            obs = self._obs[self.observation_keys[0]][indices]
            next_obs = self._next_obs[self.observation_keys[0]][indices]
        else:
            obs = np.concatenate([self._obs[k][indices] for k in
                                  self.observation_keys], axis=1)
            next_obs = np.concatenate([self._next_obs[k][indices] for k
                                       in self.observation_keys], axis=1)
        terminals = self._terminals[indices]

        batch = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
        }

        if self.sep_gripper_action:
            batch['actions'], batch['gripper_actions'] = (
                seperate_gripper_action_from_actions(
                    batch['actions'], self.gripper_idx)
            )

        return batch

    def get_transition_indices(
            self, traj_indices, k, traj_len, frame_ranges=[]):
        if k == traj_len:
            indices = np.concatenate(
                [np.arange(traj_len * i, traj_len * (i + 1))
                 for i in traj_indices])
        else:
            indices = []
            for i in traj_indices:
                traj_start_idx = traj_len * i
                traj_rand_idxs = get_rand_idxs_from_frame_ranges(
                    frame_ranges, traj_start_idx)
                indices.extend(traj_rand_idxs)
            indices = np.array(indices)

        return indices

    def random_trajectory(
            self, batch_size, with_replacement=True, k=None,
            frame_ranges=[]):
        """k = number of transitions to take from traj"""
        traj_len = self.path_len
        assert traj_len is not None and self._size % traj_len == 0
        num_traj_indices = self._size // traj_len
        traj_indices = np.random.choice(
            num_traj_indices, batch_size, replace=with_replacement)

        if k is None:
            k = traj_len
        indices = self.get_transition_indices(
            traj_indices, k, traj_len, frame_ranges)

        actions = self._actions[indices]
        rewards = self._rewards[indices]
        if len(self.observation_keys) == 1:
            obs = self._obs[self.observation_keys[0]][indices]
            next_obs = self._next_obs[self.observation_keys[0]][indices]
        else:
            obs = np.concatenate([self._obs[k][indices] for k in
                                  self.observation_keys], axis=1)
            next_obs = np.concatenate([self._next_obs[k][indices] for k
                                      in self.observation_keys], axis=1)

        terminals = self._terminals[indices]
        batch = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
        }

        for key in batch.keys():
            batch[key] = batch[key].reshape(batch_size, k, -1)
        return batch

    def _sample_goals_from_env(self, batch_size):
        return self.env.sample_goals(batch_size)

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        if self.save_data_in_snapshot:
            snapshot.update({
                'observations': self.get_slice(self._obs, slice(0, self._top)),
                'next_observations': self.get_slice(
                    self._next_obs, slice(0, self._top)
                ),
                'actions': self._actions[:self._top],
                'terminals': self._terminals[:self._top],
                'rewards': self._rewards[:self._top],
                'idx_to_future_obs_idx': (
                    self._idx_to_future_obs_idx[:self._top]
                ),
            })
        return snapshot

    def get_slice(self, obs_dict, slc):
        new_dict = {}
        for key in self.ob_keys_to_save + self.internal_keys:
            new_dict[key] = obs_dict[key][slc]
        return new_dict

    def _get_future_obs_indices(self, start_state_indices):
        future_obs_idxs = []
        for i in start_state_indices:
            possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
            lb, ub = possible_future_obs_idxs
            if ub > lb:
                next_obs_i = int(np.random.randint(lb, ub))
            else:
                next_obs_i = (
                    int(np.random.randint(lb, ub + self.max_size)) %
                    self.max_size)
            future_obs_idxs.append(next_obs_i)
        future_obs_idxs = np.array(future_obs_idxs)
        return future_obs_idxs


def combine_dicts(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: np.array([d[key] for d in dicts])
        for key in keys
    }
