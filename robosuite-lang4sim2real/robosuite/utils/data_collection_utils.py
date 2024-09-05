import argparse
from collections import Counter
import datetime
from glob import glob
import json
import os
import os.path as op

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# from rlkit.lang4sim2real_utils.lang_templates import pp_lang_by_stages_v1
import robosuite as suite
from robosuite import load_controller_config
import robosuite.utils.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING


def get_timestamp(divider='-'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider))


def create_hdf5_datasets_from_dict(grp, dic):
    for k, v in dic.items():
        grp.create_dataset(k, data=v)


def gather_demonstrations_as_hdf5(dir_list, out_dir, env_info):
    """
    This function is largely taken from:
    https://github.com/ARISE-Initiative/robosuite/blob/1b825f11a937f5c18f2ac167af8ab084275fc625/robosuite/scripts/collect_human_demonstrations.py#L83

    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        dir_list (list of strs): each element is a Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for directory in dir_list:
        for ep_directory in os.listdir(directory):

            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            # goals = []

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])

                goal = dic["goal"]

            if len(states) == 0:
                continue

            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)
            print("len(states)", len(states))

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            print("wrote goal", goal, "to dataset")
            ep_data_grp.create_dataset("goal", data=goal)

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    print("saved to", hdf5_path)
    f.close()
    return hdf5_path


def gather_demonstrations_as_hdf5_railrl_format(
        dir_list, out_dir, env_info, ep_dir_to_extras_dict_map):
    """
    This function is largely taken from:
    https://github.com/ARISE-Initiative/robosuite/blob/1b825f11a937f5c18f2ac167af8ab084275fc625/robosuite/scripts/collect_human_demonstrations.py#L83

    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        dir_list (list of strs): each element is a Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    print("dir_list", dir_list)
    # timestamp = get_timestamp()
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    task_idx_to_grp_map = dict()
    task_idx_to_num_trajs_counter = Counter()

    for directory in dir_list:
        for ep_directory in os.listdir(directory):
            ep_dir = os.path.join(directory, ep_directory)
            state_path_pattern = os.path.join(ep_dir, "state_*.npz")
            state_paths = glob(state_path_pattern)
            # There should only be 1 *.npz file under the ep_directory
            # print("state_paths", state_paths)
            assert len(state_paths) <= 1, f"files matching pattern {state_path_pattern}: {state_paths}"
            if len(state_paths) == 0:
                # traj did not succeed. skip this folder
                continue
            state_file = state_paths[0]

            dic = np.load(state_file, allow_pickle=True)
            import ipdb; ipdb.set_trace
            env_name = str(dic["env"])

            task_idx = int(dic["goal"])
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

            # obs_ep_grp = ep_data_grp.create_group("observations")
            # create_hdf5_datasets_from_dict(obs_ep_grp, {"states": dic['states']})

            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            states = np.array(dic['states'])[:-1]
            ep_data_grp.create_dataset("states", data=states)

            # process actions into (T, action_dim) np.ndarray
            # At this stage, dic['action_infos'] is a
            # List[dict[str, np.array shape (action_dim)]]
            actions_arr = np.array([ac_dict['actions'] for ac_dict in dic['action_infos']])
            assert states.shape[0] == actions_arr.shape[0]
            ep_data_grp.create_dataset("actions", data=actions_arr)
            # TODO: store sparse rewards
            # ep_data_grp.create_dataset("rewards", data=dic['rewards'])

            # TODO: support not saving next obs.
            # if "next_observations" in dic:
            #     nobs_ep_grp = ep_data_grp.create_group("next_observations")
            #     create_hdf5_datasets_from_dict(
            #         nobs_ep_grp, dic['next_observations'][()])

            # TODO: store terminals
            # ep_data_grp.create_dataset("terminals", data=dic['terminals'])

            ep_img_lang_dict = ep_dir_to_extras_dict_map[ep_dir]['img_lang_dict']
            ep_multistep_dict = ep_dir_to_extras_dict_map[ep_dir]['multistep_dict']

            env_infos_ep_grp = ep_data_grp.create_group("env_infos")
            # TODO: Save task_idx here for railrl-ut.
            create_hdf5_datasets_from_dict(
                env_infos_ep_grp, {
                    "task_idx": dic['goal'],
                    "target_obj": ep_img_lang_dict['target_obj'],
                })
            ep_data_grp.create_dataset("goal", data=dic['goal'])

            ep_data_grp.create_dataset("env", data=env_name)

            ep_data_grp.create_dataset(
                "lang_stage_num",
                data=ep_img_lang_dict['lang_stage_num'])

            if 'lang_stage_num_within_step' in ep_img_lang_dict:
                ep_data_grp.create_dataset(
                    "lang_stage_num_within_step",
                    data=ep_img_lang_dict['lang_stage_num_within_step'])
            ep_data_grp.create_dataset(
                "lang_list",
                data=ep_img_lang_dict['lang_list'])

            # Add multistep stuff
            if len(set(ep_multistep_dict['step_idx'])) > 0:
                ep_data_grp.attrs['is_multistep'] = True
                ep_data_grp.create_dataset(
                    "step_idx", data=ep_multistep_dict['step_idx'])
                ep_data_grp.create_dataset(
                    "rewards_by_step", data=ep_multistep_dict['rews_by_step'])
            else:
                ep_data_grp.attrs['is_multistep'] = False

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

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


def concat_hdf5(hdf5_list, out_dir, env_info, env_name, do_train_val_split):
    """Used to concat datasets for experimentation in robomimic"""
    out_path = os.path.join(out_dir, "combined.hdf5")
    f_out = h5py.File(out_path, mode='w')
    grp = f_out.create_group("data")

    num_eps = 1
    env_args = None
    for h5name in hdf5_list:
        h5fr = h5py.File(h5name,'r')
        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for demo_name in h5fr['data'].keys():
            h5fr.copy(f"data/{demo_name}", grp, name=f"demo_{num_eps}")
            num_eps += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    if env_args is not None:
        grp.attrs["env_args"] = env_args
    grp.attrs["env_info"] = env_info
    grp.attrs["orig_hdf5_list"] = hdf5_list

    print("saved to", out_path)
    f_out.close()

    # Make train/val split
    if do_train_val_split:
        from robomimic.scripts.split_train_val import split_train_val_from_hdf5
        split_train_val_from_hdf5(hdf5_path=out_path, val_ratio=0.1)

    return out_path


def concat_hdf5_relabeled_unique_task_idx(
        hdf5_list, out_dir, env_info, env_name, keys_to_remove=[]):

    timestamp = get_timestamp()
    out_path = os.path.join(out_dir, f"scripted_{env_name}_{timestamp}.hdf5")
    f_out = h5py.File(out_path, mode='w')
    grp = f_out.create_group("data")

    task_idx_to_num_eps_map = Counter()
    env_args = None
    buffer_source_ar = []
    old_task_idx_ar = []
    new_task_idx = -1
    env_names = []
    for i, h5name in enumerate(hdf5_list):
        h5fr = h5py.File(h5name, 'r')
        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for task_idx in h5fr['data'].keys():
            new_task_idx+=1
            buffer_source_ar.append(i)
            old_task_idx_ar.append(task_idx)

            new_task_idx = str(new_task_idx)
            task_idx_grp = grp.create_group(new_task_idx)
            task_idx = int(task_idx)
            new_task_idx = int(new_task_idx)

            task_idx_grp.attrs["old_task_idx"] = task_idx
            task_idx_grp.attrs["source_buffer"] = i
            if "env" in h5fr[f"data/{task_idx}"].attrs.keys():
                env_name = h5fr[f"data/{task_idx}"].attrs["env"]
            else:
                env_name = h5fr["data"].attrs["env"]
            task_idx_grp.attrs["env"] = env_name
            if env_name not in env_names:
                env_names.append(env_name)

            for demo_id in h5fr[f'data/{task_idx}'].keys():
                task_idx_traj_num = task_idx_to_num_eps_map[new_task_idx]
                new_name = f"demo_{task_idx_traj_num}"
                h5fr.copy(
                    f"data/{task_idx}/{demo_id}", task_idx_grp, name=new_name)
                for key_to_remove in keys_to_remove:
                    if key_to_remove in task_idx_grp[new_name].keys():
                        del task_idx_grp[f"{new_name}/{key_to_remove}"]

                task_idx_to_num_eps_map[new_task_idx] += 1
    f_out['data'].attrs['curr_idx_to_orig_buffer_src_idx_arr_map'] = (
        buffer_source_ar)
    f_out['data'].attrs['curr_to_old_task_idx_arr_map'] = old_task_idx_ar
    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["envs"] = env_names
    if env_args is not None:
        grp.attrs["env_args"] = env_args
    grp.attrs["env_info"] = env_info
    grp.attrs["orig_hdf5_list"] = hdf5_list

    print("saved to", out_path)
    f_out.close()

    return out_path


def concat_multitask_hdf5(
        hdf5_list, out_dir, env_info, env_name, do_train_val_split):
    """Used to concat datasets for experimentation in railrl"""
    out_path = os.path.join(out_dir, "combined.hdf5")
    f_out = h5py.File(out_path, mode='w')
    f_out_data_grp = f_out.create_group("data")

    # Iterate through hdf5_list and get all task_idxs in order
    # task_idxs = []
    task_idx_to_target_obj_map = {}
    for h5name in hdf5_list:
        try:
            h5fr = h5py.File(h5name, 'r')
            # test accessing data
            h5fr['data'].keys()
        except:
            # error during saving
            print(f"skipped {h5name}, bad file.")
            continue
        for task_idx in h5fr['data'].keys():
            if task_idx not in task_idx_to_target_obj_map:
                task_idx_to_target_obj_map[int(task_idx)] = h5fr[
                    f'data/{task_idx}/demo_0'].attrs['target_obj']
        # task_idxs.extend([int(task_idx) for task_idx in h5fr['data'].keys()])
    task_idxs = task_idx_to_target_obj_map.keys()

    # Create all task_idxs
    for task_idx in sorted(list(set(task_idxs))):
        f_out_data_grp.create_group(f"{task_idx}")

    # Copy trajs one by one
    task_idx_to_num_trajs_counter = Counter()
    env_args = None
    for h5name in hdf5_list:
        try:
            h5fr = h5py.File(h5name, 'r')
            # test accessing data
            h5fr['data']
        except:
            # error during saving
            print(f"skipped {h5name}, bad file.")
            continue

        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for task_idx in h5fr['data'].keys():
            task_idx = int(task_idx)
            for demo_name in h5fr[f'data/{task_idx}'].keys():
                f_out_demo_name = (
                    f"{task_idx}/demo_{task_idx_to_num_trajs_counter[task_idx]}")
                h5fr.copy(
                    f"data/{task_idx}/{demo_name}", f_out_data_grp,
                    name=f_out_demo_name)
                task_idx_to_num_trajs_counter[task_idx] += 1

                # Copy over the lang_list for each task_idx.
                if "lang_list" in h5fr[f'data/{task_idx}/{demo_name}'].keys():
                    if "lang_list" not in f_out_data_grp[f'{task_idx}'].attrs:
                        f_out_data_grp[f'{task_idx}'].attrs["lang_list"] = (
                            h5fr[f'data/{task_idx}/{demo_name}/lang_list'])
                    else:
                        demo_lang_list = np.array([
                            x.decode("ascii")
                            for x in h5fr[
                                f'data/{task_idx}/{demo_name}/lang_list'][()]])
                        assert (
                            f_out_data_grp[f'{task_idx}'].attrs["lang_list"]
                            == demo_lang_list).all()
                else:
                    if "lang_list" not in f_out_data_grp[f'{task_idx}'].attrs:
                        f_out_data_grp[f'{task_idx}'].attrs["lang_list"] = (
                            h5fr[f'data/{task_idx}'].attrs['lang_list'])
                    else:
                        demo_lang_list = h5fr[
                            f'data/{task_idx}'].attrs['lang_list'][()]
                        assert (
                            f_out_data_grp[f'{task_idx}'].attrs["lang_list"]
                            == demo_lang_list).all()

                if "lang_list" in f_out_data_grp[f_out_demo_name].keys():
                    # Delete lang_list stored in the demo, since it is
                    # already an attr of the task_idx grp
                    del f_out_data_grp[f'{f_out_demo_name}/lang_list']

                if "is_multistep" not in f_out_data_grp[f'{task_idx}'].attrs:
                    f_out_data_grp[f'{task_idx}'].attrs["is_multistep"] = h5fr[
                        f'data/{task_idx}'].attrs['is_multistep']

    # this chunk is the same as concat_hdf5
    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    f_out_data_grp.attrs["date"] = "{}-{}-{}".format(
        now.month, now.day, now.year)
    f_out_data_grp.attrs["time"] = "{}:{}:{}".format(
        now.hour, now.minute, now.second)
    f_out_data_grp.attrs["repository_version"] = suite.__version__
    f_out_data_grp.attrs["env"] = env_name
    if env_args is not None:
        f_out_data_grp.attrs["env_args"] = env_args
    f_out_data_grp.attrs["env_info"] = env_info
    f_out_data_grp.attrs["orig_hdf5_list"] = hdf5_list

    print("saved to", out_path)
    f_out.close()

    # Make train/val split
    if do_train_val_split:
        from robomimic.scripts.split_train_val import split_train_val_from_hdf5
        split_train_val_from_hdf5(hdf5_path=out_path, val_ratio=0.1)

    return out_path


def load_env_configs(args):
    # Get controller config
    controller_config = load_controller_config(
        default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Additional configs from env name
    if args.environment == "PickPlace":
        config['single_object_mode'] = 2
        config['object_type'] = "bread"

    # Check if we're using a multi-armed environment and
    # use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    return config


def post_process_hdf5(
        hdf5_path, camera_kwargs, save_next_obs, multitask_hdf5_format,
        state_mode):
    baseDir = op.abspath(
        op.join(__file__, op.pardir, op.pardir, op.pardir, op.pardir))
    prefix = "robomimic-391r"
    convert_path = os.path.join(
        baseDir, prefix, "robomimic/scripts/conversion/convert_robosuite.py")

    convert_extra_flags = ""
    if multitask_hdf5_format:
        convert_extra_flags += " --multitask-hdf5-format"

    os.system(
        f"python {convert_path} --dataset {hdf5_path} {convert_extra_flags}")

    final_dataset_path_wo_ext, ext = os.path.splitext(hdf5_path)
    final_dataset_path = f"{final_dataset_path_wo_ext}_w_obs{ext}"
    dataset_states_path = os.path.join(
        baseDir, prefix, "robomimic/scripts/dataset_states_to_obs.py")

    extra_flags = ""
    for key in camera_kwargs:
        if "camera_names" in camera_kwargs:
            extra_flags += f" --{key} {camera_kwargs[key]}"
    if save_next_obs:
        extra_flags += " --save-next-obs"
    if multitask_hdf5_format:
        extra_flags += " --multitask-hdf5-format"
    extra_flags += (
        f" --img-flip {IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]}")
    if state_mode is not None:
        extra_flags += f" --state-mode {state_mode}"

    states_to_obs_cmd = (
        f"python {dataset_states_path} --done_mode 0 --dataset {hdf5_path}"
        f" --output_name {final_dataset_path} {extra_flags}")
    print("states_to_obs_cmd", states_to_obs_cmd)
    os.system(states_to_obs_cmd)
    print("final_dataset_path", final_dataset_path)
    return final_dataset_path


def create_task_indices_from_task_int_str_list(
        task_interval_str_list, num_tasks):
    task_idx_interval_list = []
    for interval in task_interval_str_list:
        interval = tuple([int(x) for x in interval.split("-")])
        assert len(interval) == 2

        if len(task_idx_interval_list) >= 1:
            # Make sure most recently added interval's endpoint is smaller than
            # current interval's startpoint.
            assert task_idx_interval_list[-1][-1] < interval[0]

        task_idx_interval_list.append(interval)

    task_indices = []  # to collect_data on
    for interval in task_idx_interval_list:
        start, end = interval
        assert 0 <= start <= end <= num_tasks
        task_indices.extend(list(range(start, end + 1)))

    return task_indices


def save_video(out_dir, imgs):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width = imgs[0].shape[:2]
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = cv2.VideoWriter(
        os.path.join(out_dir, f"output_video{now}.mp4"),
        fourcc, 30.0, (frame_width, frame_height))
    for frame in imgs:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def save_imgs(out_dir, imgs, prefix=""):
    from PIL import Image
    for t in range(len(imgs)):
        im = Image.fromarray(imgs[t])
        prefix_new = f"{prefix}_" if prefix != "" else ""
        path = os.path.join(out_dir, f"{prefix_new}{str(t).zfill(3)}.png")
        im.save(path)


def save_demos_img_lang(out_dir, img_lang_dict_by_trajs):
    df_dict = dict(
        img_fname=[],
        lang=[],
    )

    for traj_idx, traj_dict in enumerate(img_lang_dict_by_trajs):
        imgs = traj_dict['images']
        langs = traj_dict['langs']
        assert len(imgs) == len(langs)
        for t in range(len(imgs)):
            # Save image to filename
            img = Image.fromarray(imgs[t])
            img_fname = f"{str(traj_idx).zfill(4)}_{str(t).zfill(3)}.png"
            img.save(os.path.join(out_dir, img_fname))

            # Add entry to df dict
            df_dict['img_fname'].append(img_fname)
            df_dict['lang'].append(langs[t])

    # Save df as CSV
    df = pd.DataFrame(df_dict)
    out_csv_path = os.path.join(out_dir, "labels.csv")
    df.to_csv(out_csv_path)

    return out_csv_path


def concat_img_lang_csv(img_lang_csv_paths, out_dir):
    dfs = []

    for subdir, csv_path in img_lang_csv_paths:
        try:
            df = pd.read_csv(csv_path)
            # img_dir = os.path.dirname(csv_path).split("/")[-1]
            df['img_fname'] = subdir + "/" + df['img_fname']
            dfs.append(df)
        except:
            print("failed to concat", os.path.join(subdir, csv_path))
            pass

    # Save df as CSV
    df = pd.concat(dfs, axis=0)
    out_csv_path = os.path.join(out_dir, "labels.csv")
    df.to_csv(out_csv_path)
    return out_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, default="Multitaskv1")
    parser.add_argument(
        "--robots", nargs="+", type=str, default="Panda",
        help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed",
        help="Specified environment configuration if necessary"
    )
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    args = parser.parse_args()
    hdf5_list = [
        "/home/albert/scratch/20221207/1426/1670444777_045313/demo.hdf5",
        "/home/albert/scratch/20221207/1545/1670449553_175793/demo.hdf5"]
    out_dir = "/home/albert/scratch/20221207/"
    config = load_env_configs(args)
    env_info = json.dumps(config)
    concat_hdf5(hdf5_list, out_dir, env_info)
