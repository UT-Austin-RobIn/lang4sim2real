import collections
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def l2_unit_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def expand_targets_to_pred_shape(z_preds, z_targets):
    # Assumes minibatch format for pred, where each adjacent chunk of
    # batch_size
    # items corresponds to the same task.
    if z_preds.shape[0] != z_targets.shape[0]:
        batch_size = z_preds.shape[0] // z_targets.shape[0]
        z_targets = tile_embs_by_batch_size(z_targets, batch_size)
    return z_targets


def tile_embs_by_batch_size(z_embs, batch_size):
    if isinstance(z_embs, np.ndarray):
        concat_fn = np.concatenate
        tile_fn = np.tile
    elif torch.is_tensor(z_embs):
        concat_fn = torch.cat
        tile_fn = torch.tile
    else:
        raise NotImplementedError

    z_embs = concat_fn(
        [tile_fn(z_emb, (batch_size, 1)) for z_emb in z_embs], axis=0)

    return z_embs


"""Begin: Functions Copied from robomimic/utils/tensor_utils.py,
Used for rlkit/data_management/dataset_from_hdf5.py"""


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple,
    # given a dictionary of {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary/list/tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert list not in type_func_dict
    assert tuple not in type_func_dict
    assert dict not in type_func_dict

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = (
            collections.OrderedDict() if isinstance(x, collections.OrderedDict)
            else dict())
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))


def pad_sequence_single(
        seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the time dimension (dimension 1).

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and
            end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded
            if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = (
        np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave)
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = (
        np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like)
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
        end_pad.append(repeat_func(pad, padding[1], seq_dim))

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors in the time
    dimension (dimension 1).

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or
            tuple with tensors of leading dimensions [B, T, ...]
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and
            end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if
            not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: (
                lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                    pad_sequence_single(x, p, b, ps, pv)),
            np.ndarray: (
                lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                    pad_sequence_single(x, p, b, ps, pv)),
            type(None): lambda x: x,
        }
    )


def key_is_obs_modality(key, obs_modality):
    """
    Check if observation key corresponds to modality @obs_modality.

    Args:
        key (str): obs key name to check
        obs_modality (str): observation modality - e.g.: "low_dim", "rgb"
    """
    if "image" in key:
        return obs_modality == "rgb"
    else:
        return obs_modality == "low_dim"


"""End: Functions Copied from robomimic/utils/tensor_utils.py"""


def get_rand_idxs_from_frame_ranges(frame_ranges, traj_start_idx):
    # helper function
    # Keep k frames per trajectory.
    rand_idxs = []
    for lo, hi in frame_ranges:
        rand_idx = traj_start_idx + np.random.randint(lo, hi)
        rand_idxs.append(rand_idx)
    return rand_idxs


def seperate_gripper_action_from_actions(actions, gripper_idx):
    """
    actions: a 2D array where actions[:, gripper_idx] is the gripper actions.
    """
    assert len(actions.shape) == 2
    gripper_actions = actions[:, gripper_idx]  # between [-1, 1]

    if isinstance(actions, np.ndarray):
        round_fn = np.round
        cat_fn = np.concatenate
        cat_kwargs = {"axis": 1}
    elif torch.is_tensor(actions):
        round_fn = torch.round
        cat_fn = torch.cat
        cat_kwargs = {"dim": 1}
    else:
        raise ValueError

    gripper_actions = round_fn(1 + gripper_actions)  # elements of {0, 1, 2}

    assert set(gripper_actions.tolist()).issubset(set([0., 1., 2.]))
    # 0 = close gripper
    # 1 = no change
    # 2 = open gripper
    actions = cat_fn(
        [actions[:, :gripper_idx],
         actions[:, gripper_idx + 1:]], **cat_kwargs)
    return actions, gripper_actions


def smear_traj_gripper_actions(
        traj_actions, gripper_idx, action_vals, max_path_len):
    """
    Assumes gripper actions have already been rounded to 3 action_vals
    Assumes traj_actions corresponds to actions in a single trajectory.
    """
    close_ac, neutral_ac, open_ac = action_vals
    H = max_path_len

    traj_gripper_actions = traj_actions[:, gripper_idx]
    assert traj_gripper_actions.shape == (H,)

    traj_gripper_actions = np.round(
        traj_gripper_actions).astype(int).squeeze().tolist()
    assert set(traj_gripper_actions).issubset(set(action_vals))

    do_not_smear = False
    try:
        close_idx = traj_gripper_actions.index(close_ac)
        open_idx = traj_gripper_actions.index(open_ac)
    except:
        do_not_smear = True

    if do_not_smear or close_idx >= open_idx:
        print("Ill-formed gripper trajectory; not smearing gripper action")
        # print(traj_gripper_actions)
    else:
        traj_gripper_actions = (
            [neutral_ac] * close_idx +
            [close_ac] * (open_idx - close_idx) +
            [open_ac] * (H - open_idx))

    traj_actions[:, gripper_idx] = torch.tensor(traj_gripper_actions).float()
    return traj_actions


def get_task_lang_dict_from_csv(file):
    print("Reading task language from csv file: ", file)
    map = dict()
    df = pd.read_csv(file)
    # iterate through each row in the csv
    for index, row in df.iterrows():
        if row['Task ID'] in map:
            map[int(row['Task ID'])].append(row["Instruction"])
        else:
            map[int(row['Task ID'])] = [row['Instruction']]
    
    list = []
    for i in range(max(map) + 1):
        if i not in map:
            list.append([])
        else:  
            list.append(map[i])

    return list


    
