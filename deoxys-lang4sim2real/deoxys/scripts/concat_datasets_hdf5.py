from deoxys.utils.data_collection_utils import (
    load_env_info, concat_hdf5,
    maybe_create_data_save_path)
import argparse
from glob import glob
import os


if __name__ == "__main__":
    # Sample usage:
    # Concatenate all under current dir with */scripted* pattern
    # python concat_datasets_hdf5.py --save-orig-hdf5-list -e frka_obj_bowl_plate -d /home/robin/Projects/albert/datasets/realrobot/2hz/carrot_cont_plate -p /home/robin/Projects/albert/datasets/realrobot/2hz/carrot_cont_plate/ --prefix */scripted
    # Concatenate all hdf5 within a directory
    # python concat_datasets_hdf5.py --save-orig-hdf5-list -e frka_obj_bowl_plate -d /home/robin/Projects/albert/datasets/realrobot/2hz/carrot_cont_plate/2024-01-05T22-23-02/ -p /home/robin/Projects/albert/datasets/realrobot/2hz/carrot_cont_plate/2024-01-05T22-23-02/
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-p", "--hdfs", nargs="+", type=str, default=[])
    parser.add_argument("-d", "--save-directory", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--save-orig-hdf5-list", action="store_true", default=False)
    args = parser.parse_args()

    if len(args.hdfs) > 1:
        assert args.prefix == "", "No support for prefix yet"
        hdf5_paths = args.hdfs
        out_dir = args.save_directory
    elif len(args.hdfs) == 1:
        # Search for *.hdf5 files in args.save_directory, concatenate them all.
        data_save_path = maybe_create_data_save_path(args.save_directory)
        thread_outpaths = os.path.join(args.hdfs[0], f"{args.prefix}*.hdf5")
        hdf5_paths = list(glob(thread_outpaths))
        print("len(hdf5_paths)", len(hdf5_paths))
        print("hdf5_paths", hdf5_paths)
        out_dir = data_save_path
    elif len(args.hdfs) == 0:
        # Search for *.hdf5 files in args.save_directory, concatenate them all.
        data_save_path = maybe_create_data_save_path(args.save_directory)
        thread_outpaths = os.path.join(data_save_path, f"{args.prefix}*.hdf5")
        hdf5_paths = list(glob(thread_outpaths))
        out_dir = data_save_path
    else:
        print("Nothing to concatenate")

    print("hdf5_paths", hdf5_paths)
    env_info = load_env_info(args.env_name)
    concat_hdf5(
        hdf5_paths, out_dir, env_info, args.env_name,
        save_orig_hdf5_list=args.save_orig_hdf5_list,
        demo_attrs_to_del=["lang_list", "is_multistep"])

    # Clean up tmp files
    tmp_dir = f"{args.save_directory}/tmp/"
    if os.path.exists(tmp_dir):
        os.system(f"rm -r {tmp_dir}")
        print(f"Removed {tmp_dir}")
