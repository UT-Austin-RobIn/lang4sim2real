"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.

Based on robomimic/utils/dataset.py
"""
import os
import time

from contextlib import contextmanager
import h5py
import numpy as np
import torch.utils.data as torchdata

import rlkit.util.misc_functions as miscFuncs


class MultitaskSequenceDataset(torchdata.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        task_idx_to_seq_length_map={},
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        filter_by_attribute=None,
        del_cache_cleanup=True,
        cache_demos_for_enc=False,
        demo_enc_frame_ranges=None,
        success_only=True,
        max_num_demos_per_task=None,
        task_to_max_num_demos_override_map={},
        task_idxs=[],
        gripper_smear_ds_actions=False,
        gripper_idx=None,
        max_path_len=30,
        expand_action_dim_params={"task_idxs": []},
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to
        (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items
                (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items
                (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch.
                Defaults to 1 (single frame).

            task_idx_to_seq_length_map (Dict[int, int]): maps each task_idx to
                length of sequences to sample. 1 == single frame.

            pad_frame_stack (int): whether to pad sequence for frame stacking
                at the beginning of a demo. This ensures that partial frame
                stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise,
                the first frame stacked observation would be
                (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching
                at the end of a demo. This ensures that partial sequences at
                the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence
                provided would be (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of
                the batch. This can be useful for masking loss functions on
                padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is
                to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to
                "all" to cache entire hdf5 in memory - this is by far the
                fastest for data loading. Set to "low_dim" to cache all 
                non-image data. Set to None to use no caching - in this case,
                every batch sample is retrieved via file i/o. You should almost
                never set this to None, even for large image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening
                the hdf5 file. This ensures that multiple Dataset instances can
                all access the same hdf5 file without problems.

            filter_by_attribute (str): if provided, use the provided filter key
                to look up a subset of
                demonstrations to load

            cache_demos_for_enc (bool): whether to store all demos for
                demo encoder in memory

            demo_enc_frame_ranges (list of len-2 int tuples):
                ex: [(0, 1), (29, 30)].
                Only used when cache_demos_for_enc == True

            success_only (bool): only save trajectories that end in
                positive reward

            max_num_demos_per_task (int or None): if None, no limit on trajs
                per task. if int, this is the limit.

            task_to_max_num_demos_override_map:
                (dict[task_idx(int), num_demos(int)])
                for each task_idx in this dict, override max_num_demos_per_task
                with num_demos.

            task_idxs (list): Only gather trajs from hdf5 if they have
                task idxs in this list.

            gripper_smear_ds_actions (bool): whether or not to smear
                gripper actions.

            gripper_idx (int): index of action space controlling gripper.

            expand_action_dim_params (dict):
                - task_idxs: List[int] of task idxs to expand action dims.
                - expand_idx: (int) add an extra action dim between
                              action[:expand_idx] and action[expand_idx]
                - expand_size: (int) how many dims to add
        """
        super(MultitaskSequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.filter_by_attribute = filter_by_attribute

        self.cache_demos_for_enc = cache_demos_for_enc
        self.demo_enc_frame_ranges = demo_enc_frame_ranges
        self.success_only = success_only
        self.max_num_demos_per_task = max_num_demos_per_task
        self.task_to_max_num_demos_override_map = (
            task_to_max_num_demos_override_map)
        self.task_idxs = task_idxs

        self.gripper_smear_ds_actions = gripper_smear_ds_actions
        self.gripper_idx = gripper_idx
        self.expand_action_dim_params = expand_action_dim_params
        self.expand_action_dims = bool(
            len(self.expand_action_dim_params["task_idxs"])
            and {"expand_idx", "expand_size"}.issubset(
                self.expand_action_dim_params.keys()))

        self.max_path_len = max_path_len

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.task_idx_to_seq_length_map = task_idx_to_seq_length_map
        assert all([
            seq_length >= 1
            for task_idx, seq_length in task_idx_to_seq_length_map.items()])

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        # maybe store dataset in memory for fast access
        self.maybe_set_lang_desc_attrs()
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if miscFuncs.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache, self.vid_enc_demos_cache = (
                self.load_dataset_in_memory(
                    demo_list=self.demos,
                    hdf5_file=self.hdf5_file,
                    obs_keys=self.obs_keys_in_memory,
                    dataset_keys=self.dataset_keys,
                ))

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this
                # for "low-dim" since image observations require calls to
                # getitem anyways.
                print("MultitaskSequenceDataset: caching get_item calls...")
                self.getitem_cache = [
                    self.get_item(i)
                    for i in miscFuncs.custom_tqdm(range(len(self)))]

                if del_cache_cleanup:
                    # don't need the previous cache anymore
                    del self.hdf5_cache
                    self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def maybe_set_lang_desc_attrs(self):
        if "lang_list" in self.hdf5_file[f'data/{self.task_idxs[0]}'].attrs:
            self.task_idx_to_lang_idx_offset = {}
            self.lang_desc_list = []
            self.include_lang_idx_in_batch = True
            for task_idx in self.task_idxs:
                task_lang_desc_list = (
                    self.hdf5_file[f'data/{task_idx}'].attrs['lang_list'])
                self.task_idx_to_lang_idx_offset[task_idx] = len(
                    self.lang_desc_list)
                self.lang_desc_list.extend(task_lang_desc_list)

    def load_demo_info(self, filter_by_attribute=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load
        """
        hdf5_task_idxs = sorted(
            [int(task_idx) for task_idx in self.hdf5_file["data"].keys()])
        assert set(self.task_idxs).issubset(set(hdf5_task_idxs))

        if set(self.task_idxs) != set(hdf5_task_idxs):
            print(
                f"filtering buffer from {sorted(hdf5_task_idxs)}\n"
                f"to -->\n{sorted(self.task_idxs)}")

        self.task_idx_to_demo_id_list_map = {}
        self.task_idx_to_n_demos_map = {}
        self.demos = []
        for task_idx in self.task_idxs:
            task_idx_demo_ids = list(self.hdf5_file[f"data/{task_idx}"].keys())
            traj_inds = np.argsort(
                [int(demo_id[5:]) for demo_id in task_idx_demo_ids])

            # Maybe filter out non-successful trajectories.
            task_idx_demo_ids_to_keep = []
            task_idx_max_num_demos_to_keep = (
                self.task_to_max_num_demos_override_map.get(
                    task_idx, self.max_num_demos_per_task))
            for i in traj_inds:
                if (task_idx_max_num_demos_to_keep is not None and
                        (len(task_idx_demo_ids_to_keep) >=
                            task_idx_max_num_demos_to_keep)):
                    print(f"truncating hdf5 buffer task ID {task_idx} ...")
                    break
                demo_id = task_idx_demo_ids[i]
                
                traj_is_successful = self.hdf5_file[
                    f"data/{task_idx}/{demo_id}/rewards"][-1] > 0

                success_sat = ((not self.success_only) or
                               (self.success_only and traj_is_successful))
                filter_sat = ((filter_by_attribute is None) or
                              (f"{task_idx}{demo_id}".encode()
                               in self.hdf5_file[
                                f'mask/{filter_by_attribute}']))
                keep_traj = success_sat and filter_sat
                if keep_traj:
                    task_idx_demo_ids_to_keep.append(demo_id)
                else:
                    print("traj not kept")

            self.task_idx_to_demo_id_list_map[task_idx] = (
                task_idx_demo_ids_to_keep)
            self.demos.extend([
                f"{task_idx}/{demo_id}"
                for demo_id in task_idx_demo_ids_to_keep])
            self.task_idx_to_n_demos_map[task_idx] = len(
                task_idx_demo_ids_to_keep)

        self.n_demos = len(self.demos)
        print(self.task_idx_to_n_demos_map)
        print("Total num demos", self.n_demos)

        # keep internal index maps to know which transitions
        # belong to which demos
        self._index_to_demo_id = dict()
        # maps every transition index to a demo id {0: "0/demo_0"}
        self._demo_id_to_start_indices = dict()
        self._demo_id_to_demo_length = dict()
        self._task_idx_to_ds_idxs = dict()  # maps each task idx to a list of
        # transition indices in self._index_to_demo_id.keys()

        # determine index mapping
        self.total_num_sequences = 0
        for task_idx in self.task_idxs:
            self._task_idx_to_ds_idxs[task_idx] = []
            for ep in self.task_idx_to_demo_id_list_map[task_idx]:
                try:
                    demo_length = self.hdf5_file[
                        f"data/{task_idx}/{ep}"].attrs["num_samples"]
                except:
                    demo_length = self.hdf5_file[
                        f"data/{task_idx}/{ep}/actions"][()].shape[0]
                    # self.max_path_len
                self._demo_id_to_start_indices[f"{task_idx}/{ep}"] = (
                    self.total_num_sequences)
                self._demo_id_to_demo_length[f"{task_idx}/{ep}"] = demo_length

                num_sequences = demo_length
                # determine actual number of sequences taking into account
                # whether to pad for frame_stack and seq_length
                if not self.pad_frame_stack:
                    num_sequences -= (self.n_frame_stack - 1)
                if not self.pad_seq_length:
                    num_sequences -= (
                        self.task_idx_to_seq_length_map[task_idx] - 1)

                if self.pad_seq_length:
                    assert demo_length >= 1, (
                        "sequence needs to have at least one sample")
                    num_sequences = max(num_sequences, 1)
                else:
                    assert num_sequences >= 1

                for _ in range(num_sequences):
                    self._index_to_demo_id[self.total_num_sequences] = (
                        f"{task_idx}/{ep}")
                    self._task_idx_to_ds_idxs[task_idx].append(
                        self.total_num_sequences)
                    self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(
                self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        print("self._hdf5_file", self._hdf5_file)
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except:
                print("hdf5 file failed to close")
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n"
        msg += "\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = (
            self.filter_by_attribute if self.filter_by_attribute is not None
            else "none")
        goal_mode_str = (
            self.goal_mode if self.goal_mode is not None else "none")
        cache_mode_str = (
            self.hdf5_cache_mode if self.hdf5_cache_mode is not None
            else "none")
        msg = msg.format(self.hdf5_path, self.obs_keys,
                         self.task_idx_to_seq_length_map,
                         filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack,
                         goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through
        all sequences in the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(
            self, demo_list, hdf5_file, obs_keys, dataset_keys):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the
        file. Note that this differs from `self.getitem_cache`, which, if
        active, actually caches the outputs of the `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        vid_enc_demos = dict()
        print("MultitaskSequenceDataset: loading dataset into memory...")
        for ep in miscFuncs.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            try:
                all_data[ep]["attrs"]["num_samples"] = hdf5_file[
                    "data/{}".format(ep)].attrs["num_samples"]
            except:
                all_data[ep]["attrs"]["num_samples"] = self.max_path_len
            # get obs
            all_data[ep]["observations"] = {
                k: hdf5_file[
                    "data/{}/observations/{}".format(ep, k)][()].astype(
                        'float32')
                for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file[
                        "data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    try:
                        all_data[ep][k] = np.zeros(
                            (all_data[ep]["attrs"]["num_samples"], 1),
                            dtype=np.float32)
                    except:
                        all_data[ep][k] = np.zeros(
                            (self.max_path_len, 1), dtype=np.float32)

                if k == "lang_idx" and self.include_lang_idx_in_batch:
                    task_idx = int(ep.split("/")[0])
                    lang_idx_offset = (
                        self.task_idx_to_lang_idx_offset[task_idx])
                    all_data[ep][k] = (
                        lang_idx_offset
                        + hdf5_file[f"data/{ep}/lang_stage_num"][()])

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file[
                    "data/{}".format(ep)].attrs["model_file"]

            # Maybe Load demo array for demo encoder into cache
            if self.cache_demos_for_enc:
                transition_rand_idxs = (
                    miscFuncs.get_rand_idxs_from_frame_ranges(
                        self.demo_enc_frame_ranges, traj_start_idx=0))
                vid_enc_demos[ep] = hdf5_file[f"data/{ep}/observations/image"][
                    transition_rand_idxs]

        return all_data, vid_enc_demos

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert key1 in ['observations', 'next_observations']
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert key1 in ['observations', 'next_observations']
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            # print("hd5key", hd5key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map),
        using the getitem_cache if available.
        """
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_task_idx = int(demo_id.split("/")[0])
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = (
            0 if self.pad_frame_stack
            else (self.n_frame_stack - 1))
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = (
            0 if self.pad_seq_length
            else (self.task_idx_to_seq_length_map[demo_task_idx] - 1))
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.task_idx_to_seq_length_map[demo_task_idx]
        )

        # Process the action
        if self.gripper_smear_ds_actions:
            meta["actions"] = miscFuncs.smear_traj_gripper_actions(
                meta["actions"], self.gripper_idx, (-1, 0, 1),
                self.max_path_len)
        if demo_task_idx in self.expand_action_dim_params['task_idxs']:
            expand_idx = self.expand_action_dim_params['expand_idx']
            expand_size = self.expand_action_dim_params['expand_size']
            meta["actions"] = np.concatenate(
                [meta["actions"][:, :expand_idx],
                 np.zeros((meta["actions"].shape[0], expand_size)),
                 meta["actions"][:, expand_idx:]], axis=1)

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["observations"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.task_idx_to_seq_length_map[demo_task_idx],
            prefix="observations"
        )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_observations",
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}
            # remove sequence dimension for goal

        return meta

    def get_sequence_from_demo(
            self, demo_id, index_in_demo, keys, num_frames_to_stack=0,
            seq_length=1, sample_size=None):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of
        the items.

        Args:
            demo_id (str): task_id and id of the demo, e.g., "0/demo_0"
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets
                prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended
                with repeated items if out of range
            sample_size (int or None): downsample (sub)sequence to sample_size
                number of evenly spaced transitions.

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)
        # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)
        # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            if sample_size is not None:
                assert seq_begin_index == 0, (
                    "start sampling from beginning of traj.")
                assert sample_size <= seq_end_index - seq_begin_index - 1
                sampled_idxs = np.round(np.linspace(
                    seq_begin_index, seq_end_index - 1, sample_size)
                ).astype(int)
                seq[k] = data[sampled_idxs].astype("float32")
            else:
                seq[k] = data[seq_begin_index: seq_end_index].astype("float32")

        seq = miscFuncs.pad_sequence(
            seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array(
            [0] * seq_begin_pad +
            [1] * (seq_end_index - seq_begin_index) +
            [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(np.bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(
            self, demo_id, index_in_demo, keys,
            num_frames_to_stack=0, seq_length=1, prefix="observations",
            sample_size=None):
        """
        Extract a (sub)sequence of observation items from a demo given the
        @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets
                prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended
                with repeated items if out of range
            prefix (str): one of "obs", "next_obs"
            sample_size (int or None): downsample (sub)sequence to sample_size
                number of evenly spaced transitions.

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
            sample_size=sample_size,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        # return process_obs_dict(obs)
        return obs

    def get_dataset_sequence_from_demo(
            self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of
        the items (e.g., states, actions).
        
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended
                with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_traj_with_demo_id(self, demo_id):
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["observations"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )

        meta["ep"] = demo_id
        return meta

    def get_traj_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        return self.get_traj_with_demo_id(demo_id)

    def get_task_rand_traj(self, task_idx):
        demo_id = np.random.choice(self.task_idx_to_demo_id_list_map[task_idx])
        demo_id = f"{task_idx}/{demo_id}"
        return self.get_traj_with_demo_id(demo_id)

    def get_task_idx_rand_demo_for_enc(self, task_idx):
        demo_id = np.random.choice(self.task_idx_to_demo_id_list_map[task_idx])
        demo_id = f"{task_idx}/{demo_id}"
        assert self.cache_demos_for_enc
        demo_for_enc = self.vid_enc_demos_cache[demo_id]
        # returns uint8 (len(frame_ranges), 48, 48, 3).
        # Convert to float
        if demo_for_enc.dtype == "uint8":
            demo_for_enc = demo_for_enc / 255.0
        else:
            print("warning, unexpected dtype for demo image from cache")
        return demo_for_enc

    def get_task_idx_dataset_sampler(self, task_idx):
        """
        Return instance of torchdata.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        sampler = torchdata.SubsetRandomSampler(
            self._task_idx_to_ds_idxs[task_idx])
        return sampler


class MultitaskDataLoader:
    """Inits a DataLoader for each task_idx to support getting a batch"""
    def __init__(self, dataset, batch_size, num_data_workers):
        self.task_idx_to_dataloaders = dict()
        self.task_idx_to_dataloader_iters = dict()
        self.dataset = dataset
        for task_idx in dataset.task_idxs:
            task_idx_dataloader = torchdata.DataLoader(
                dataset=dataset,
                sampler=dataset.get_task_idx_dataset_sampler(task_idx),
                batch_size=(
                    batch_size[task_idx] if isinstance(batch_size, dict)
                    else batch_size),
                num_workers=num_data_workers,
                drop_last=True,
                persistent_workers=False,
                # Make `persistent_workers` true if using num_workers > 1.
                # Makes huge speed difference
            )
            self.task_idx_to_dataloaders[task_idx] = task_idx_dataloader
            self.task_idx_to_dataloader_iters[task_idx] = iter(
                task_idx_dataloader)

    def get_task_idx_batch(self, task_idx):
        try:
            task_idx_dataloader_iter = self.get_task_dataloader(task_idx)
            batch = next(task_idx_dataloader_iter)
        except StopIteration:
            task_idx_dataloader_iter = self.reset_task_dataloader_iter(
                task_idx)
            batch = next(task_idx_dataloader_iter)
        # import ipdb; ipdb.set_trace()
        return batch

    def get_task_idx_rand_traj(self, task_idx):
        # Returns a single random trajectory of task_idx
        return self.dataset.get_task_rand_traj(task_idx)

    def get_task_dataloader(self, task_idx):
        return self.task_idx_to_dataloader_iters[task_idx]

    def reset_task_dataloader_iter(self, task_idx):
        dataloader = self.task_idx_to_dataloaders[task_idx]
        self.task_idx_to_dataloader_iters[task_idx] = iter(dataloader)
        return self.task_idx_to_dataloader_iters[task_idx]


if __name__ == "__main__":
    hdf5_path = "/home/albert/dev/minibullet-ut/data/data/demo.hdf5"
    meta_bsz = 16
    bsz = 16
    ds = MultitaskSequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=["state", "image"],
        dataset_keys=["actions"],
        hdf5_cache_mode="low_dim",
    )
    print("1 ds._hdf5_file", ds._hdf5_file)
    dl = MultitaskDataLoader(ds, bsz, num_data_workers=0)
    print("2 ds._hdf5_file", ds._hdf5_file)
    for task_idx in ds.task_idxs:
        s = time.time()
        dl.get_task_idx_batch(task_idx)
        print(f"time to load bsz={bsz}", time.time() - s)
        print("3 ds._hdf5_file", ds._hdf5_file)

    time.sleep(1)
    print("closing hdf5 file")
    del dl
