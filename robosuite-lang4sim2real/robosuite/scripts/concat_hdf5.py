import argparse
import json
import os
from robosuite.utils.data_collection_utils import (
    concat_multitask_hdf5,
    concat_hdf5_relabeled_unique_task_idx)
from glob import glob


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_env_info(env_name):
    config = {"env_name": env_name}
    env_info = json.dumps(config)
    return env_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env-names", nargs='+', type=str, required=True)
    parser.add_argument(
        "-p", "--hdfs", nargs="+", type=str, default=[])
    parser.add_argument(
        "-d", "--save-directory", type=str, required=True)
    parser.add_argument(
        "--remove-next-obs", action="store_true", default=False)
    parser.add_argument(
        "--concat-mode", choices=["merge-on-task-idx", "relabel-task-idx"],
        default="merge-on-task-idx")
    args = parser.parse_args()

    if len(args.hdfs) >= 1:
        hdf5_paths = args.hdfs
        out_dir = maybe_create_dir(args.save_directory)
        if len(args.hdfs) == 1:
            assert args.remove_next_obs
            input(
                "You only have one dataset to concatenate, are you sure? "
                "CTRL+C to quit.")
    elif len(args.hdfs) == 0:
        # Search for *.hdf5 files in args.save_directory, concatenate them all.
        data_save_path = maybe_create_dir(args.save_directory)
        thread_outpaths = os.path.join(data_save_path, "*.hdf5")
        hdf5_paths = list(glob(thread_outpaths))
        out_dir = data_save_path

    print("hdf5_paths", hdf5_paths)
    env_info = load_env_info(args.env_names[0])
    keys_to_remove = []
    if args.remove_next_obs:
        keys_to_remove.append("next_observations")

    kwargs = {}
    if args.concat_mode == "merge-on-task-idx":
        concat_fn = concat_multitask_hdf5
    elif args.concat_mode == "relabel-task-idx":
        concat_fn = concat_hdf5_relabeled_unique_task_idx
    concat_fn(
        hdf5_paths,
        out_dir,
        env_info,
        args.env_names[0],
        keys_to_remove,
        **kwargs)

    # Clean up tmp files
    tmp_dir = f"{args.save_directory}/tmp/"
    if os.path.exists(tmp_dir):
        os.system(f"rm -r {tmp_dir}")
        print(f"Removed {tmp_dir}")
