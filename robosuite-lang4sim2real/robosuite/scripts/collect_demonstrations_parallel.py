import argparse
import datetime
import json
import os
import subprocess
import time


import numpy as np

from robosuite.utils.data_collection_utils import (
    load_env_configs, concat_hdf5, concat_multitask_hdf5, concat_img_lang_csv)
from robosuite.policies import NAME_TO_POLICY_CLASS_MAP


def get_timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider))


def get_data_save_dir(args):
    data_save_directory = args.directory

    data_save_directory += '_{}'.format(args.environment)

    if args.num_trajs > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajs/1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajs)

    data_save_directory += '_{}'.format(get_timestamp())

    return data_save_directory


def convert_args_to_command_list(args, blacklist):
    dirname = os.path.dirname(os.path.abspath(__file__))
    command_list = ['python', dirname + '/collect_demonstrations.py']
    for k, v in args.__dict__.items():
        if k in blacklist:
            continue

        # replace _ --> -
        k = k.replace("_", "-")

        if isinstance(v, list):
            flag = [f"--{k}"] + v
        elif v is True:
            flag = [f"--{k}"]
        elif v is False:
            # Don't add v=false flags since flags take value True when added.
            flag = []
        else:
            flag = [f"--{k}", f"{v}"]
        command_list.extend(flag)
    return command_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-trajs", type=int, required=True)
    parser.add_argument("-p", "--num-threads", type=int, required=True)

    parser.add_argument("--directory", type=str, required=True)
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
    parser.add_argument("--device", type=str, default="scripted-policy")
    parser.add_argument("--save-img-lang-pairs", action="store_true", default=False)
    parser.add_argument("--policy", type=str, default="", choices=list(NAME_TO_POLICY_CLASS_MAP.keys()))
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--task-idx-intervals", nargs="+", type=str, default="")
    parser.add_argument("-t", "--max-horiz-len", type=int, default=0, help="How many maximum timesteps to allow for each trajectory.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="How much gaussian noise to add to scripted policy.")
    parser.add_argument("--save-next-obs", action="store_true", default=False, help="whether or not to save next_obs.")
    parser.add_argument(
        "--multitask-hdf5-format", action="store_true", default=False,
        help="True = compatible with railrl. False = compatible with robomimic.")
    parser.add_argument("--intra-thread-delay", default=2, type=int)
    parser.add_argument(
        "--state-mode", type=int, default=None, choices=[0, 1, 2, None], help="For Multitaskv2 or MultitaskMM, what mode to use for creating the state vector.")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--randomize", type=str, default="", choices=["wide", "narrow", ""])
    parser.add_argument(
        "--adr-rna", default=False, action="store_true",
        help=("Use this flag to collect data for the baseline:"
              " Automatic Domain Rando (ADR) and"
              " Random Network Adversary (RNA)."))
    args = parser.parse_args()

    # make a new timestamped directory for output
    out_dir = os.path.join(args.directory, f"combined_{get_timestamp()}")
    assert not os.path.exists(out_dir)
    os.makedirs(out_dir)

    num_trajs_per_thread = int(np.ceil(
        args.num_trajs / args.num_threads))

    save_dir = get_data_save_dir(args)

    command_list = convert_args_to_command_list(
        args, blacklist=["num_threads", "num_trajs", "intra_thread_delay"])
    command_list.extend(["--num-trajs", str(num_trajs_per_thread)])
    print(command_list)

    subprocesses = []
    tmp_dirs = []
    hdf5_paths = []
    img_lang_csv_paths = []
    for i in range(args.num_threads):
        # Additional args
        tmp_dir = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        tmp_dirs.append(tmp_dir)
        command_list.extend(["--tmp-directory", tmp_dir])

        # Set subdir
        t1, t2 = str(time.time()).split(".")
        subdir = "{}_{}".format(t1, t2)
        command_list.extend(["--subdir", subdir])

        # Add postprocess flag
        if not args.save_img_lang_pairs:
            command_list.extend(["--post-process"])

        if args.save_img_lang_pairs:
            csv_path = os.path.join(args.directory, subdir, "labels.csv")
            img_lang_csv_paths.append((subdir, csv_path))
        else:
            # Keep list of hdf5_paths
            hdf5_path = os.path.join(args.directory, subdir, "demo_w_obs.hdf5")
            # hdf5_path = post_process_hdf5(hdf5_path)
            hdf5_paths.append(hdf5_path)

        subprocesses.append(subprocess.Popen(command_list))
        time.sleep(args.intra_thread_delay)

    exit_codes = [p.wait() for p in subprocesses]

    # Get EnvInfo
    config = load_env_configs(args)
    env_info = json.dumps(config)

    if not args.save_img_lang_pairs:
        # Concat all hdf5s

        if args.multitask_hdf5_format:
            out_path = concat_multitask_hdf5(
                hdf5_paths, out_dir, env_info, args.environment,
                do_train_val_split=False)
        else:
            out_path = concat_hdf5(
                hdf5_paths, out_dir, env_info, args.environment,
                do_train_val_split=True)
        print("final dataset", out_path)
    else:
        out_path = concat_img_lang_csv(img_lang_csv_paths, args.directory)
