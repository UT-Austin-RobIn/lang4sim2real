import argparse
import h5py
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hdf", type=str, default="")
    args = parser.parse_args()

    h5fr = h5py.File(args.hdf, 'r')

    out_dir, in_fname = os.path.split(args.hdf)
    in_fname_wo_ext = os.path.splitext(in_fname)[0]

    out_path = os.path.join(out_dir, f"{in_fname_wo_ext}_postprocessed.hdf5")
    copy_cmd = f"cp {args.hdf} {out_path}"
    os.system(copy_cmd)
    f_out = h5py.File(out_path, mode='r+')

    # Rename keys
    # obs --> observations
    # next_obs --> next_observations
    for task_idx in tqdm(f_out['data'].keys()):
        for demo_id in f_out[f'data/{task_idx}'].keys():
            f_out.move(
                f"data/{task_idx}/{demo_id}/obs",
                f"data/{task_idx}/{demo_id}/observations")
            f_out.move(
                f"data/{task_idx}/{demo_id}/next_obs",
                f"data/{task_idx}/{demo_id}/next_observations")

    # Add num samples
    # TODO
