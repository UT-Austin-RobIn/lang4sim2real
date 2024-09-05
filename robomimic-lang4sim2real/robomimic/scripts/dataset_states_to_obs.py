"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0], f"{states.shape[0]} != {actions.shape[0]}"

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t], "goal": initial_state["goal"]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_states_to_obs(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
        state_mode=args.state_mode,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")

    if args.multitask_hdf5_format:
        task_idx_strs = [task_idx for task_idx in f['data'].keys()]
        sorted_task_idxs = np.argsort(task_idx_strs)
        demos = []
        for task_idx_idx in sorted_task_idxs:
             task_idx = task_idx_strs[task_idx_idx]
             task_idx_demos = list(f[f'data/{task_idx}'].keys())
             inds = np.argsort([int(elem[5:]) for elem in task_idx_demos])
             demos.extend([f"{task_idx}/{task_idx_demos[i]}" for i in inds])
    else:
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

    # set hdf5 key strs
    assert len(args.camera_names) <= 1
    if args.multitask_hdf5_format:
        key_str_map = dict(
            obs="observations",
            next_obs="next_observations",
            image="image",
            task_id="one_hot_task_id",
            dones="terminals",
        )
    else:
        key_str_map = dict(
            obs="obs",
            next_obs="next_obs",
            image=args.camera_names[0] if len(args.camera_names) == 1 else "",
            task_id="task_id",
            dones="dones",
        )

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind, ep in enumerate(demos):

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        goal = f["data/{}/goal".format(ep)][()]
        initial_state = dict(
            states=states[0],
            goal=goal)
        if is_robosuite_env:
            initial_state["model"] = f[
                "data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset(
            "lang_stage_num",
            data=np.array(f[f"data/{ep}/lang_stage_num"][()]).astype(np.int32))
        ep_data_grp.create_dataset("lang_list", data=f[f"data/{ep}/lang_list"])
        if f[f"data/{ep}"].attrs["is_multistep"]:
            ep_data_grp.create_dataset(
                "step_idx",
                data=np.array(f[f"data/{ep}/step_idx"][()]).astype(np.int32))
            ep_data_grp.create_dataset(
                "rewards_by_step",
                data=np.array(f[f"data/{ep}/rewards_by_step"][()]))
            ep_data_grp.create_dataset(
                "lang_stage_num_within_step",
                data=np.array(
                    f[f"data/{ep}/lang_stage_num_within_step"][()]
                ).astype(np.int32))

        # Set is_multistep attr under the task_idx
        if len(ep.split("/")) == 2:
            task_idx = ep.split("/")[0]
            if ("is_multistep" not in data_grp[f"{task_idx}"].attrs
                    and "is_multistep" in f[f"data/{ep}"].attrs):
                data_grp[f"{task_idx}"].attrs["is_multistep"] = f[
                    f"data/{ep}"].attrs["is_multistep"]

        if args.multitask_hdf5_format:
            dones = np.array(traj["dones"], dtype=bool)
        else:
            dones = np.array(traj["dones"])
        ep_data_grp.create_dataset(f"{key_str_map['dones']}", data=dones)

        for k in traj["obs"]:
            # print("task_id sum", np.sum(traj['obs']['task_id']))
            k_key_str = k
            obs_data = np.array(traj["obs"][k])
            nobs_data = np.array(traj["next_obs"][k])
            if "image" in k:
                k_key_str = key_str_map["image"]
                # obs['...image...'] is (T, H, W, C). We want to flip H dim.
                obs_data = obs_data[:, ::args.img_flip]
                nobs_data = nobs_data[:, ::args.img_flip]
            elif k == "task_id":
                k_key_str = key_str_map["task_id"]
            ep_data_grp.create_dataset(
                f"{key_str_map['obs']}/{k_key_str}", data=obs_data)
            if args.save_next_obs:
                ep_data_grp.create_dataset(
                    f"{key_str_map['next_obs']}/{k_key_str}", data=nobs_data)

        if args.multitask_hdf5_format:
            # add a state observation key to include all other stuff
            print("env.env.state_keys", env.env.state_keys)
            state_data = np.concatenate([
                traj["obs"][state_obs_key]
                if len(traj["obs"][state_obs_key].shape) > 1
                else traj["obs"][state_obs_key][:, None]
                for state_obs_key in env.env.state_keys], axis=-1)
            ep_data_grp.create_dataset(
                f"{key_str_map['obs']}/state", data=state_data)

            # add env_infos and task_idx
            ep_data_grp.create_dataset(
                "env_infos/task_idx",
                data=np.array([goal] * state_data.shape[0]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj[
                "initial_state_dict"]["model"]  # model xml for this episode
            ep_data_grp.attrs["target_obj"] = f[
                f"data/{ep}/env_infos/target_obj"][()].decode('ascii')
        ep_data_grp.attrs["num_samples"] = (
            traj["actions"].shape[0])  # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(
            ind, ep_data_grp.attrs["num_samples"], ep))

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4)  # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--multitask-hdf5-format",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-next-obs",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--img-flip",
        default=1,  # no flip
        type=int,
        choices=[1, -1],  # -1 = flip
    )
    parser.add_argument(
        "--state-mode",
        default=None,
        type=int,
        choices=[0, 1, 2, None],
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
