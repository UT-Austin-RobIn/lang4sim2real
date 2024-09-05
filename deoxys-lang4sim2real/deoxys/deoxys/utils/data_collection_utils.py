from collections import Counter
import os
import os.path as osp
import sys

import datetime
from glob import glob
import h5py
import json
import numpy as np
import pygame
from pygame import QUIT, MOUSEBUTTONDOWN
from tqdm import tqdm

import deoxys.utils.calibration_utils as cal_utils
from deoxys.utils.obj_detector import ObjectDetectorDL
from deoxys.utils.params import *


def get_timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider))


def postprocess_traj_dict_for_hdf5(traj):
    assert isinstance(traj, dict)

    def flatten_env_infos(traj):
        flattened_env_infos = {}
        for env_info_key in traj['env_infos'][0]:
            if type(traj['env_infos'][0][env_info_key]) in [int, bool]:
                flattened_env_infos[env_info_key] = np.array(
                    [traj['env_infos'][i][env_info_key]
                     for i in range(len(traj['env_infos']))])
        return flattened_env_infos

    def flatten_obs(traj_obs):
        flattened_obs = {}
        for key in traj_obs[0].keys():
            flattened_obs[key] = np.array([
                traj_obs[i][key] for i in range(len(traj_obs))])
        return flattened_obs

    orig_traj_keys = list(traj.keys())
    flattened_keys = set(['actions', 'rewards', 'terminals'])
    keys_to_remove = set(['agent_infos'])
    keys_to_flatten = sorted(
        list(set(traj.keys()) - flattened_keys - keys_to_remove))

    for key in orig_traj_keys:
        if key in keys_to_remove:
            traj.pop(key, None)
        elif key in keys_to_flatten:
            if key in ["observations", "next_observations"]:
                traj[key] = flatten_obs(traj[key])
            elif key == "env_infos":
                traj['env_infos'] = flatten_env_infos(traj)
            else:
                raise NotImplementedError
        else:
            assert key in flattened_keys

    return traj


def create_hdf5_datasets_from_dict(grp, dic):
    for k, v in dic.items():
        grp.create_dataset(k, data=v)


def gather_demonstrations_as_hdf5(dir_list, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during
            collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        dir_list (list of strs): each element is a Path to the directory
            containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    print("dir_list", dir_list)
    timestamp = get_timestamp()
    hdf5_path = os.path.join(out_dir, f"{timestamp}.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    task_idx_to_grp_map = dict()
    task_idx_to_num_trajs_counter = Counter()

    for directory in dir_list:
        for ep_directory in os.listdir(directory):

            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            # There should only be 1 *.npz file under the ep_directory
            assert len(glob(state_paths)) == 1
            state_file = glob(state_paths)[0]

            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            task_idx = dic["env_infos"][()]["task_idx"][0]
            # print("task_idx", task_idx)
            task_idx_traj_num = task_idx_to_num_trajs_counter[task_idx]
            if task_idx_traj_num == 0:
                task_grp = grp.create_group(f"{task_idx}")
                task_idx_to_grp_map[task_idx] = task_grp
            else:
                task_grp = task_idx_to_grp_map[task_idx]

            ep_data_grp = task_grp.create_group(
                "demo_{}".format(task_idx_traj_num))

            # write datasets for all items in the trajectory.

            obs_ep_grp = ep_data_grp.create_group("observations")
            create_hdf5_datasets_from_dict(obs_ep_grp, dic['observations'][()])

            ep_data_grp.create_dataset("actions", data=dic['actions'])
            ep_data_grp.create_dataset("rewards", data=dic['rewards'])

            if "next_observations" in dic:
                nobs_ep_grp = ep_data_grp.create_group("next_observations")
                create_hdf5_datasets_from_dict(
                    nobs_ep_grp, dic['next_observations'][()])

            ep_data_grp.create_dataset("terminals", data=dic['terminals'])

            env_infos_ep_grp = ep_data_grp.create_group("env_infos")
            create_hdf5_datasets_from_dict(
                env_infos_ep_grp, dic['env_infos'][()])

            ep_data_grp.create_dataset("env", data=env_name)

            n_sample = ep_data_grp["actions"].shape[0]
            ep_data_grp.attrs["num_samples"] = n_sample

            num_eps += 1
            task_idx_to_num_trajs_counter[task_idx] += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    print("saved to", hdf5_path)
    f.close()
    return hdf5_path


def concat_hdf5(
        hdf5_list, out_dir, env_info, env_name,
        save_orig_hdf5_list=False, demo_attrs_to_del=[]):
    timestamp = get_timestamp()
    out_path = os.path.join(out_dir, f"scripted_{env_name}_{timestamp}.hdf5")
    f_out = h5py.File(out_path, mode='w')
    grp = f_out.create_group("data")

    task_idx_to_num_eps_map = Counter()
    env_args = None
    for h5name in tqdm(hdf5_list):
        print(h5name)
        h5fr = h5py.File(h5name, 'r')
        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for task_idx in h5fr['data'].keys():
            if task_idx not in f_out['data'].keys():
                task_idx_grp = grp.create_group(task_idx)
            else:
                task_idx_grp = f_out[f'data/{task_idx}']
            task_idx = int(task_idx)
            for demo_id in h5fr[f'data/{task_idx}'].keys():
                # Set lang_list under task_idx grp
                if "lang_list" not in task_idx_grp.attrs.keys():
                    if "lang_list" in h5fr[f'data/{task_idx}'].attrs.keys():
                        task_idx_grp.attrs["lang_list"] = (
                            h5fr[f'data/{task_idx}'].attrs["lang_list"])
                    else:
                        task_idx_grp.attrs["lang_list"] = [
                            str(x) for x in (
                                h5fr[f'data/{task_idx}/{demo_id}']
                                .attrs['lang_list'])]
                if "is_multistep" not in task_idx_grp.attrs.keys():
                    task_idx_grp.attrs["is_multistep"] = h5fr[
                        f'data/{task_idx}'].attrs["is_multistep"]

                task_idx_traj_num = task_idx_to_num_eps_map[task_idx]
                h5fr.copy(
                    f"data/{task_idx}/{demo_id}", task_idx_grp,
                    name=f"demo_{task_idx_traj_num}")

                for attr_to_del in demo_attrs_to_del:
                    if attr_to_del in (
                            task_idx_grp[f'demo_{task_idx_traj_num}']
                            .attrs.keys()):
                        del (task_idx_grp[f"demo_{task_idx_traj_num}"]
                             .attrs[attr_to_del])
                task_idx_to_num_eps_map[task_idx] += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    if env_args is not None:
        grp.attrs["env_args"] = env_args
    grp.attrs["env_info"] = env_info
    print("env_info", env_info)
    if save_orig_hdf5_list:
        hdf5_list_dict = json.dumps(dict(
            zip(range(len(hdf5_list)), sorted(hdf5_list))))
        grp.attrs["orig_hdf5_list"] = hdf5_list_dict

    print("saved to", out_path)
    f_out.close()

    return out_path


def load_env_info(env_name):
    config = {"env_name": env_name}
    env_info = json.dumps(config)
    return env_info


def maybe_create_data_save_path(save_directory):
    data_save_path = osp.join(__file__, "../..", "data", save_directory)
    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)
    return data_save_path


def paint_pp_rewards(env, T, successful_traj, gripper_states_arr):
    """
    Paints pick and place rewards retroactively
    gripper_states_arr: is from next_obs
    T: time horizon of trajectory
    """
    if successful_traj:
        # Find the idx where it goes from closed to open.
        # Use next_obs to determine which timestep the close/open occurred.
        gripper_open_list = list(env.gripper_open(gripper_states_arr))
        try:
            gripper_close_ts = gripper_open_list.index(0)
            gripper_open_ts = gripper_open_list[gripper_close_ts:].index(1)
            success_ts = gripper_close_ts + gripper_open_ts
            reward_vec = np.array([0.] * (success_ts) + [1.] * (T - success_ts))
        except:
            reward_vec = np.zeros((T,))
            print(
                "Detected obj in successful place but unable to paint "
                "rewards retroactively.")

    print("rewards", reward_vec)
    return reward_vec


def get_obj_xy_pos(env, obj_name, lift_before_reset=False, conf_thresh=None):
    # Gets obj pos in robot xy coords
    if hasattr(env, "obj_detector") and env.obj_detector is not None:
        obj_detector = env.obj_detector
        wait_secs = 0.01
    else:
        obj_detector = ObjectDetectorDL(env=env)
        wait_secs = 1.0
    img = obj_detector.get_img(
        transpose=False, lift_before_reset=lift_before_reset,
        wait_secs=wait_secs)
    centroids = obj_detector.get_centroids(img, conf_thresh=conf_thresh)
    obj_pos_in_robot_coords = cal_utils.rgb_to_robot_coords(
        centroids[obj_name], DL_RGB_TO_ROBOT_TRANSMATRIX)
    return np.squeeze(obj_pos_in_robot_coords)  # (1D or 2D array)


def get_objs_bbox(env, lift_before_reset=False, fmt="xyxy", wait_secs=1.0):
    if hasattr(env, "obj_detector") and env.obj_detector is not None:
        obj_detector = env.obj_detector
    else:
        obj_detector = ObjectDetectorDL(env=env, skip_reset=True)
    img = obj_detector.get_img(
        transpose=False, lift_before_reset=lift_before_reset,
        wait_secs=wait_secs)
    bboxes = obj_detector.get_bounding_box(img)
    # format is {obj_name: (x_min, y_min, x_max, y_max)}
    if fmt == "xyxy":
        return bboxes
    elif fmt == "vertex_list":
        for obj_name, xyxy in bboxes.items():
            x_min, y_min, x_max, y_max = bboxes[obj_name]
            bboxes[obj_name] = [
                (x_min, y_min),
                (x_min, y_max),
                (x_max, y_max),
                (x_max, y_min),
                (x_min, y_min)]
    else:
        raise NotImplementedError
    return bboxes


def is_camera_feed_live(
        obs1, obs2, im_diff_thresh=0.001,
        state_diff_thresh=0.0):
    """
    If norm(action) >= act_norm_thresh, then check that the im1 and im2 differ
    by at least im_diff_thresh proprtion of array elements.

    returns True when camera feed is alive, False when the feed is frozen.
    """
    def get_(obs_dict, key):
        """
        if we are doing multiple substeps per step, then everything in
        obs_dict is a list
        """
        if isinstance(obs_dict[key], list):
            return obs_dict[key][-1]
        else:
            return obs_dict[key]

    im1 = get_(obs1, "image")
    im2 = get_(obs2, "image")
    state1 = np.concatenate(
        [get_(obs1, "ee_pos"), get_(obs1, "gripper_states")])  # (4,)
    state2 = np.concatenate(
        [get_(obs2, "ee_pos"), get_(obs2, "gripper_states")])  # (4,)
    assert im1.dtype == im2.dtype == "uint8"
    assert im1.shape == im2.shape

    num_diff_pixels_thresh = int(im_diff_thresh * np.prod(im1.shape))
    # ^ ~49 for 128x128x3
    num_diff_pixels = np.sum(im1 != im2)
    print("num_diff_pixels", num_diff_pixels)

    if np.linalg.norm(state1 - state2) > state_diff_thresh:
        im1_and_im2_diff_enough = (
            num_diff_pixels > num_diff_pixels_thresh)
        return im1_and_im2_diff_enough
    else:
        # action was too small, don't expect images to be different
        return True


class PyGameListener:
    def __init__(self):
        pygame.init()
        canvas = pygame.display.set_mode((200, 200), 0, 32)
        red = (255, 0, 0)
        canvas.fill(red)
        pygame.display.update()

    def quit(self):
        pygame.quit()
        sys.exit()

    def check_mouse_click(self):
        any_mouse_click = False
        for event in pygame.event.get():
            if event.type == QUIT:
                run = False

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 == left button
                    any_mouse_click = True
        return any_mouse_click

    def clear_events(self):
        pygame.event.clear()
