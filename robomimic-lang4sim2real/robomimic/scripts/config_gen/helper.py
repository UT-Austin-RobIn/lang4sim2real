import argparse
import os
import time
import datetime

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

def get_generator(algo_name, config_file, args, algo_name_short=None):
    if args.wandb_proj_name is None:
        strings = [
            algo_name_short if (algo_name_short is not None) else algo_name,
            args.name,
            args.env,
            args.mod
        ]
        args.wandb_proj_name = '_'.join([s for s in strings if s is not None])

    if args.script is not None:
        generated_config_dir = os.path.join(os.path.dirname(args.script), "json")
    else:
        curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%y-%H-%M-%S')
        generated_config_dir=os.path.join(
            os.getenv("HOME"), 'tmp/autogen_configs/lcil', algo_name, args.env, args.mod, args.name, curr_time, "json",
        )

    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=generated_config_dir,
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    return generator

def set_debug_mode(generator, args):
    if not args.debug:
        return

    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.horizon",
        name="",
        group=-1,
        values=[100],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.epochs",
        name="",
        group=-1,
        values=[None],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.name",
        name="",
        group=-1,
        values=["debug"],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=[None],
        value_names=[""],
    )

def set_exp_id(generator, args):
    assert args.name is not None

    vals = generator.parameters["train.output_dir"].values

    for i in range(len(vals)):
        vals[i] = os.path.join(vals[i], args.name)

def set_num_rollouts(generator, args):
    if args.nr < 0:
        return
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[args.nr],
        value_names=[""],
    )

def set_wandb_mode(generator, args):
    if args.no_wandb:
        generator.add_param(
            key="experiment.logging.log_wandb",
            name="",
            group=-1,
            values=[False],
        )

def set_video_mode(generator, args):
    if args.no_video:
        generator.add_param(
            key="experiment.render_video",
            name="",
            group=-1,
            values=[False],
        )
        generator.add_param(
            key="experiment.keep_all_videos",
            name="",
            group=-1,
            values=[False],
        )
    else:
        generator.add_param(
            key="experiment.keep_all_videos",
            name="",
            group=-1,
            values=[True],
        )

def set_env_settings(generator, args):
    if args.env == 'calvin':
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            if args.mod == 'ld':
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["scene_obs", "robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "scene_proprio",
                    ],
                    hidename=True,
                )
            else:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "proprio",
                    ],
                    hidename=True,
                )

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[["rgb_static"]],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[["rgb_gripper"]],
            )

        generator.add_param(
            key="train.load_next_obs",
            name="",
            group=-1,
            values=[False],
        )
        if "experiment.rollout.horizon" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[600],
            )
    elif args.env == 'kitchen':
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["flat"]
            ],
        )
        if args.mod == 'im':
            raise NotImplementedError
    elif args.env in ['square', 'lift']:
        # set videos off
        args.no_video = True

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "object"]
                ],
            )
    elif args.env == 'transport':
        # set videos off
        args.no_video = True

        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["robot0_eef_pos",
                 "robot0_eef_quat",
                 "robot0_gripper_qpos",
                 "robot1_eef_pos",
                 "robot1_eef_quat",
                 "robot1_gripper_qpos",
                 "object"],
            ],
        )
        if args.mod == 'im':
            raise NotImplementedError
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )
    else:
        raise ValueError

def set_mod_settings(generator, args):
    if args.mod == 'ld':
        generator.add_param(
            key="experiment.rollout.epochs",
            name="",
            group=-1,
            values=[
                [50, 100, 150] + [100*i for i in range(2, 21)],
            ],
        )
        generator.add_param( # adding this bc caching the sequence dataset for lang embeddings takes too much memory
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            values=["low_dim"],
        )
    elif args.mod == 'im':
        generator.add_param(
            key="experiment.save.epochs",
            name="",
            group=-1,
            values=[
                [200, 400, 600]
            ],
        )
        generator.add_param(
            key="experiment.epoch_every_n_steps",
            name="",
            group=-1,
            values=[500],
        )
        generator.add_param(
            key="train.num_data_workers",
            name="",
            group=-1,
            values=[2],
        )
        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            values=["low_dim"],
        )
        generator.add_param(
            key="train.batch_size",
            name="",
            group=-1,
            values=[16],
        )
        generator.add_param(
            key="train.num_epochs",
            name="",
            group=-1,
            values=[600],
        )
        generator.add_param(
            key="experiment.rollout.rate",
            name="",
            group=-1,
            values=[20],
        )


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--env",
        type=str,
        required=True,
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        required=True
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        '--no_video',
        action='store_true'
    )

    parser.add_argument(
        "--tmplog",
        action="store_true",
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    return parser

def make_generator(args, make_generator_helper):
    if args.tmplog or args.debug:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + args.name

    if args.debug:
        args.no_wandb = True

    if args.wandb_proj_name is not None:
        # prepend data to wandb name
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.wandb_proj_name = time_str + args.wandb_proj_name

    if (args.debug or args.tmplog) and (args.wandb_proj_name is None):
        args.wandb_proj_name = 'debug'

    # make config generator
    generator = make_generator_helper(args)

    set_env_settings(generator, args)
    set_mod_settings(generator, args)
    set_debug_mode(generator, args)
    set_num_rollouts(generator, args)
    set_exp_id(generator, args)
    set_wandb_mode(generator, args)
    set_video_mode(generator, args)

    generator.add_param(
        key="experiment.save.on_best_rollout_success_rate",
        name="",
        group=-1,
        values=[
            False
        ],
    )

    # generate jsons and script
    generator.generate()
