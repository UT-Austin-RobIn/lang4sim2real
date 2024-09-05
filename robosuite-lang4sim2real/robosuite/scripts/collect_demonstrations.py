"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
import json
import os
import time
from collections import Counter

import cv2
import datetime
import numpy as np
from tqdm import tqdm

import robosuite as suite
from robosuite.utils.data_collection_utils import (
    gather_demonstrations_as_hdf5_railrl_format,
    load_env_configs, create_task_indices_from_task_int_str_list,
    save_demos_img_lang, save_video, save_imgs)
from robosuite.wrappers import (
    AutomaticDomainRandomizationWrapper,
    DataCollectionWrapper,
    VisualizationWrapper)
from robosuite.policies import NAME_TO_POLICY_CLASS_MAP


def collect_scripted_trajectory(env, policy, args):
    """
    Use a scripted policy to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        policy (Policy): to receive actions from the Policy given state
        args (dict): other parameters.
    """

    retry = True
    num_attempts = 0
    while retry:
        img_lang_dict = dict(
            images=[],
            lang_list=[],
            lang_stage_num=[],
            lang_stage_num_within_step=[],
            target_obj="",
        )
        multistep_dict = dict(
            rews_by_step=[],
            step_idx=[],
            step_idx_to_num_stages_map=[],
        )
        mm_dict = dict()

        obs = env.reset(save_traj=(not args.success_only))
        # HACKY fix, there's some bug where this env.reset() results in 2 _reset_internal() calls but returns the state after the first resulting in an invalid randomly selected object.
        # So we force the observation to be after the second reset environment
        # obs, _, _, _ = env.step([0,0,0,0,0,0,0])

        # ID = 2 always corresponds to agentview
        # if args.gui: env.render()

        update_lang_list = False

        if env.num_steps > 1:
            # Need to reset to start-traj policy kwargs (reset step_idx related target obj/conts)
            policy, args = maybe_update_scripted_policy_kwargs(policy, args, env)

        policy.start_control()

        # Loop until we get a reset from the input or the task completes
        for t in range(args.max_horiz_len):
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp, policy_info = scriptedPolicy2action(
                policy=policy, env_obs=obs, robot=active_robot,
                active_arm=args.arm, env_configuration=args.config
            )

            assert action is not None

            action += np.random.normal(loc=0.0, scale=args.noise_std)

            if env.num_steps > 1:
                prev_step_idx = env.step_idx

            # Run environment step
            obs, reward, done, info = env.step(action)
            # except Exception as e:
            #     save_imgs("error_traj", img_lang_dict['images'])
            #     exit()
            # print(f"t = {t} step time: {time.time() - s}")
            t += 1

            if args.save_img_lang_pairs or args.save_video or args.debug:
                im = cv2.resize(obs['agentview_image'], (128, 128))
                if args.save_video:
                    im = cv2.resize(obs['agentview_image'], (256, 256))
                img_lang_dict['images'].append(im)

            if args.save_video and t == args.max_horiz_len - 1:
                if args.save_video == 1:
                    save_video("out_dir_imgs", img_lang_dict['images'])
                elif args.save_video == 2:
                    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_imgs("out_dir_imgs", img_lang_dict['images'], now)

            if env.num_steps > 1:
                policy = maybe_update_scripted_policy_multistep(policy, env, prev_step_idx)
                multistep_dict['rews_by_step'].append(info['rews_by_step'])
                multistep_dict['step_idx'].append(info['step_idx'])
                if update_lang_list:
                    # Wait an extra timestep after step_idx increments so that
                    # lang_list will update properly.
                    img_lang_dict['lang_list'].extend(
                        policy_info['policy_lang_list'])
                    assert len(img_lang_dict['lang_list']) == np.sum(
                        multistep_dict['step_idx_to_num_stages_map'])
                    update_lang_list = False

                # calculate multistep lang_stage_num (across the different steps)
                cum_num_stages = np.sum(multistep_dict['step_idx_to_num_stages_map'][:-1])
                img_lang_dict['lang_stage_num'].append(
                    cum_num_stages + policy_info['policy_lang_stage_num'])
                img_lang_dict['lang_stage_num_within_step'].append(policy_info['policy_lang_stage_num'])

                if info['step_idx'] == len(multistep_dict['step_idx_to_num_stages_map']):
                    multistep_dict['step_idx_to_num_stages_map'].append(policy_info['total_num_stages'])
                    update_lang_list = True
            else:
                img_lang_dict['lang_stage_num'].append(policy_info['policy_lang_stage_num'])
                img_lang_dict['lang_list'] = policy_info['policy_lang_list']

            if args.gui:
                env.render()

        # Last success
        success = bool(reward > 0.0)

        # cleanup for end of data collection episodes
        if not args.success_only or (args.success_only and success):
            # No more retries.
            retry = False
            ep_dir = env.ep_directory
            img_lang_dict['target_obj'] = env.get_obj_str(env.object_id)
            env.close(save_traj=True)
        else:
            # Dont save if we need to retry.
            env.close(save_traj=False)

        print("num_attempts", num_attempts, "retry", retry)
        # Automatic Domain Rando
        if args.adr_rna:
            env._post_traj(reward)

        num_attempts += 1

    extras_dict = dict(
        img_lang_dict=img_lang_dict,
        multistep_dict=multistep_dict,
        mm_dict=mm_dict,
    )

    return success, num_attempts, extras_dict, ep_dir


def get_scripted_policy_kwargs(
        env, policy_name, obj_name=None, cont_name=None, drop_pos_offset=None):
    # Common kwargs to all envs
    policy_kwargs = {
        'eef_pos_obs_key': 'robot0_eef_pos',
        'pos_sensitivity': 0.05,
        'pick_z_thresh': 0.97,
        'env': env,
    }

    policy_name_to_extra_kwargs_map = {
        "grasp": {
            "target_obj_obs_key": "cube_pos",
            "horiz_len": 100,
        },
        "pick_place": {
            "target_obj_obs_key": "Bread_pos",
            "horiz_len": 200,
        },
        "stack": {
            "target_obj_obs_key": "Bread_pos",
            "stack": True,
            "stack_obj_obs_key": f"{cont_name}_pos",
            "horiz_len": 200,
        },
        "push": {
            "target_obj_obs_key": "Bread_pos",
            "target_dst_obs_key": "zone_pos",
            "horiz_len": 200,
        },
        "wrap": {
            "target_geom_obs_key": "spoolandwire_geom_13_pos",
            # "target_geom_obs_key": "spoolandwire_geom_0_pos",
            "center_geom_key": "spoolandwire_geom_0_pos",
            "horiz_len": 400,
            "pick_z_thresh": .85,
        },
        "wrap-completion": {
            "target_geom_obs_key": "spoolandwire_geom_12_pos",
            # "target_geom_obs_key": "spoolandwire_geom_0_pos",
            "center_geom_key": "spool_geom_0_pos",
            "horiz_len": 200,
            "pick_z_thresh": .85,
            "relative_location_lang": False,
        },
        "wrap-relative-location": {
            "target_geom_obs_key": "spoolandwire_geom_12_pos",
            # "target_geom_obs_key": "spoolandwire_geom_0_pos",
            "center_geom_key": "spool_geom_0_pos",
            "horiz_len": 200,
            "pick_z_thresh": .85,
            "relative_location_lang": True,
        },

        # Manner of Motion Scripted Policies
        ("stack", "quickly"): {
            "target_obj_obs_key": "Bread_pos",
            "stack": True,
            "stack_obj_obs_key": f"{cont_name}_pos",
            "mm_kwargs": dict(
                action_coeff=3.5,
            ),
        },
        ("stack", "slowly"): {
            "target_obj_obs_key": "Bread_pos",
            "stack": True,
            "stack_obj_obs_key": f"{cont_name}_pos",
            "mm_kwargs": dict(
                action_coeff=0.8,
            ),
        }
    }

    extra_kwargs = dict(policy_name_to_extra_kwargs_map[policy_name])

    if env.is_multitask_env or "wrap" in policy_name:
        if "wrap" not in policy_name:
            max_horiz_len = env.get_horizon_from_task_id()
            extra_kwargs.pop("horiz_len")
        else:
            max_horiz_len = extra_kwargs.pop("horiz_len")
        if (obj_name is not None
                and env.is_multitask_env
                and "wrap" not in policy_name):
            extra_kwargs["target_obj_obs_key"] = f"{obj_name}_pos"
        extra_kwargs["drop_pos_offset"] = drop_pos_offset

    if env.num_steps > 1:
        max_horiz_len = 0
        for reward_mode in env.step_kwargs["reward_modes"]:
            step_horiz_len = (
                policy_name_to_extra_kwargs_map[reward_mode]["horiz_len"])
            max_horiz_len += step_horiz_len
        max_horiz_len *= 0.8
        max_horiz_len = int(10 * (max_horiz_len // 10))

    policy_kwargs.update(extra_kwargs)
    return policy_kwargs, max_horiz_len


def maybe_update_scripted_policy_kwargs(device, args, env):
    if args.device != "scripted-policy":
        return device, args

    if env.is_multitask_env:
        args.policy = env.get_reward_mode()
    else:
        assert args.policy != ""

    obj_name = env.id_to_object[env.object_id]

    if env.num_steps > 1:
        # assumes this function is called before start of each new trajectory
        # sometimes env.step_idx is still at 1 from the previous traj,
        # so we directly index into offset 0 instead of env.step_idx.
        drop_pos_offset = env.step_kwargs["drop_pos_offsets"][0]
    else:
        drop_pos_offset = None

    policy_kwargs, args.max_horiz_len = get_scripted_policy_kwargs(
        env, args.policy, obj_name=obj_name, cont_name=env.stack_cont_name,
        drop_pos_offset=drop_pos_offset)

    if isinstance(args.policy, tuple):
        args.policy = args.policy[0]
    policy_class = NAME_TO_POLICY_CLASS_MAP[args.policy]
    policy = policy_class(**policy_kwargs)

    if env.is_multitask_env and "wrap" not in args.policy:
        policy.target_obj = obj_name.lower()

    return policy, args


def maybe_update_scripted_policy_multistep(policy, env, prev_step_idx):
    """
    Only switch policies if env has advanced
    """
    if prev_step_idx == env.step_idx:
        return policy

    # Else, initialize new policy

    policy_name = env.step_kwargs['reward_modes'][env.step_idx]
    policy_class = NAME_TO_POLICY_CLASS_MAP[policy_name]
    obj_name = env.step_kwargs['obj_names'][env.step_idx]
    if obj_name == "<target_obj>":
        assert env.is_multitask_env
        obj_name = env.obj_names[env.object_id]
    cont_name = env.step_kwargs['cont_names'][env.step_idx]
    drop_pos_offset = env.step_kwargs['drop_pos_offsets'][env.step_idx]

    # get policy_kwargs with updated obj_name and cont_name
    policy_kwargs, _ = get_scripted_policy_kwargs(
        env, policy_name, obj_name, cont_name, drop_pos_offset)
    policy = policy_class(**policy_kwargs)
    if env.is_multitask_env and "wrap" not in policy_name:
        policy.target_obj = obj_name  # env.id_to_object[env.object_id].lower()
    policy.start_control()
    return policy


def collect_human_trajectory(env, device, args):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        args (dict): other parameters.
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    # if args.gui: env.render()

    task_completion_hold_count = -1
    # ^ counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    t = 0  # num timesteps
    while True:
        # Set active robot
        active_robot = (
            env.robots[0] if args.config == "bimanual"
            else env.robots[args.arm == "left"])

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm,
            env_configuration=args.config
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        action += args.noise_std

        # Run environment step
        env.step(action)
        t += 1

        if args.gui:
            env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        if t >= args.max_horiz_len:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--subdir", type=str, default="") # Set by collect_demonstrations_parallel.py
    parser.add_argument("--tmp-directory", type=str, default="")
    parser.add_argument("--post-process", action="store_true", default=False)
    parser.add_argument("-e", "--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--img-dim", type=int, default=0, help="Dimension of image obs to save.")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse", "scripted-policy"])
    parser.add_argument("--save-img-lang-pairs", action="store_true", default=False)
    parser.add_argument("--save-video", type=int, choices=[0, 1, 2], default=0)
    # 0 = no video saving. 1 = save video. 2 = save image frames individually.
    parser.add_argument("--policy", type=str, default=None, choices=list(NAME_TO_POLICY_CLASS_MAP.keys()) + [""])
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("-n", "--num-trajs", type=int, default=100)
    parser.add_argument("--task-idx-intervals", nargs="+", type=str, default="")
    parser.add_argument("-t", "--max-horiz-len", type=int, default=None, help="How many maximum timesteps to allow for each trajectory.")
    # parser.add_argument("-m", "--num-tasks", type=int, default=None, help="Only used for multitask envs.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="How much gaussian noise to add to scripted policy.")
    parser.add_argument("--gui", action="store_true", default=False, help="Whether or not to turn on GUI")
    parser.add_argument("--save-next-obs", action="store_true", default=False, help="whether or not to save next_obs.")
    parser.add_argument(
        "--multitask-hdf5-format", action="store_true", default=False,
        help="True = compatible with railrl. False = compatible with robomimic.")
    parser.add_argument(
        "--state-mode", type=int, default=None, choices=[0, 1, 2, None], help="For Multitaskv2 or MultitaskMM, what mode to use for creating the state vector.")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether or not to turn on debug mode.")
    parser.add_argument("--randomize", type=str, default="", choices=["wide", "narrow", ""])
    parser.add_argument(
        "--adr-rna", default=False, action="store_true",
        help=("Use this flag to collect data for the baseline:"
              " Automatic Domain Rando (ADR) and"
              " Random Network Adversary (RNA)."))
    args = parser.parse_args()

    config = load_env_configs(args)

    kwargs = dict(
        has_renderer=args.gui,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )
    if args.randomize:
        kwargs['randomize'] = args.randomize

    if args.state_mode is not None:
        kwargs['state_mode'] = args.state_mode

    if args.save_img_lang_pairs or args.img_dim > 0 or args.save_video:
        kwargs.update(dict(
            has_offscreen_renderer=True,
            use_camera_obs=True,
        ))
    camera_kwargs = {}
    if args.save_img_lang_pairs or args.save_video:
        kwargs.update(dict(camera_names=["agentview", "frontview"]))
    elif args.img_dim > 0:
        camera_kwargs = dict(
            camera_names=args.camera,
            camera_height=args.img_dim,
            camera_width=args.img_dim,
        )
        kwargs.update(dict(
            camera_names=args.camera,
            image_dim=args.img_dim,
        ))

    # Create environment
    env = suite.make(
        **config,
        **kwargs,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Turn off the green line in GUI
    env.set_visualization_setting(setting="grippers", visible=False)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    if args.tmp_directory == "":
        args.tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))

    # If doing Domain randomization, use wrapper
    if args.adr_rna:
        env = AutomaticDomainRandomizationWrapper(
            env=env,
            param_step_size=.1,
            num_trajs_per_bound_update=4)
    env = DataCollectionWrapper(env, args.tmp_directory, allow_prints=False)

    print(
        "env.get_task_lang_dict()['instructs']",
        env.get_task_lang_dict()['instructs'])

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        from robosuite.utils.input_utils import input2action

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
        collect_fn = collect_human_trajectory
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        from robosuite.utils.input_utils import input2action

        device = SpaceMouse(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity)
        collect_fn = collect_human_trajectory
    elif args.device == "scripted-policy":
        from robosuite.utils.input_utils import scriptedPolicy2action
        collect_fn = collect_scripted_trajectory

        args.success_only = True

        if env.is_multitask_env:
            if not args.task_idx_intervals:
                task_idxs_list = list(range(env.num_tasks))
            else:
                task_idxs_list = create_task_indices_from_task_int_str_list(
                    args.task_idx_intervals, env.num_tasks)
            assert args.num_trajs % len(task_idxs_list) == 0
            print("task_idxs_list", task_idxs_list)
            num_trajs_per_task = args.num_trajs // len(task_idxs_list)

        device = None
        # ^ will be set in maybe_update_scripted_policy_kwargs(...)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard'"
            "or 'spacemouse' or 'scripted-policy'.")

    # make a new timestamped directory
    if args.subdir == "":
        t1, t2 = str(time.time()).split(".")
        args.subdir = "{}_{}".format(t1, t2)
    new_dir = os.path.join(args.directory, args.subdir)
    os.makedirs(new_dir)

    # collect demonstrations
    num_demos = 0
    task_id = -1
    num_successes_by_task_id = Counter()
    num_trajs_by_task_id = Counter()
    # img_lang_dict_by_trajs = []
    # ep_dir_to_img_lang_dict_map = {}
    extras_dict_by_trajs = []
    ep_dir_to_extras_dict_map = {}
    for i in tqdm(range(args.num_trajs)):
        if args.device == "scripted-policy" and env.is_multitask_env:
            task_id = task_idxs_list[i // num_trajs_per_task]
            env.set_task_id(task_id)
        device, args = maybe_update_scripted_policy_kwargs(
            device, args, env)

        # Only save sucessful trajectories.
        success, num_attempts, extras_dict, ep_dir = collect_fn(
            env, device, args)
        # img_lang_dict_by_trajs.append(img_lang_dict)
        # ep_dir_to_img_lang_dict_map[ep_dir] = img_lang_dict
        extras_dict_by_trajs.append(extras_dict)
        ep_dir_to_extras_dict_map[ep_dir] = extras_dict
        num_successes_by_task_id[task_id] += success
        num_trajs_by_task_id[task_id] += num_attempts

    if args.save_img_lang_pairs:
        out_path = save_demos_img_lang(
            new_dir, ep_dir_to_extras_dict_map['img_lang_dict'])
    else:
        hdf5_path = gather_demonstrations_as_hdf5_railrl_format(
            [args.tmp_directory], new_dir, env_info, ep_dir_to_extras_dict_map)

    # Compute success rates
    success_rates_dict = dict(
        [(task_id, (
            num_successes_by_task_id[task_id]
            / num_trajs_by_task_id[task_id]))
         for task_id in num_successes_by_task_id])
    print("Success rates by task", success_rates_dict)

    if args.post_process:
        from robosuite.utils.data_collection_utils import post_process_hdf5
        hdf5_path = post_process_hdf5(
            hdf5_path, camera_kwargs, args.save_next_obs,
            args.multitask_hdf5_format, args.state_mode)  # demo_w_obs.hdf5
