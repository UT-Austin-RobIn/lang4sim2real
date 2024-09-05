import argparse
import os
from PIL import Image
import sys

import h5py
import nexusformat.nexus as nx
import numpy as np

from deoxys.envs import init_env


def create_rew_im_bar(rews, film_strip_arr_shape):
    assert len(film_strip_arr_shape) == 3
    assert film_strip_arr_shape[1] % rews.shape[0] == 0
    assert len(rews.shape) == 1
    rew_bar_ts_h, rew_bar_ts_w = (
        film_strip_arr_shape[0] // 10,
        film_strip_arr_shape[1] // rews.shape[0])
    rew_im_bar = []
    for rew in rews:
        rew_bar_ts = np.zeros([rew_bar_ts_h, rew_bar_ts_w, 3], dtype=np.uint8)
        if rew > 0:
            rew_bar_ts[:, :] = [0, 255, 0]
        else:
            rew_bar_ts[:, :] = [255, 0, 0]
        rew_im_bar.append(rew_bar_ts)

    return np.concatenate(rew_im_bar, axis=1)


def save_film_strips(args):
    out_dir = args.film_strip_out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with h5py.File(args.path, mode='r') as h5fr:
        for task_id in h5fr['data'].keys():
            for demo_id in h5fr[f'data/{task_id}'].keys():
                # Example of loading attributes
                # Visualize film strip
                ims = h5fr[f"data/{task_id}/{demo_id}/observations/image"][()]
                film_strip_arr = np.concatenate(list(ims), axis=1)

                rews = h5fr[f"data/{task_id}/{demo_id}/rewards"]
                rew_im_bar = create_rew_im_bar(rews, film_strip_arr.shape)
                film_strip_arr = np.concatenate(
                    [film_strip_arr, rew_im_bar], axis=0)

                film_strip_im = Image.fromarray(film_strip_arr)
                out_fpath = os.path.join(
                    out_dir, f"task_{task_id}_{demo_id}.png")
                film_strip_im.save(out_fpath)
                print(h5fr[f'data/{task_id}/{demo_id}/actions'][()])
                # quit()


def save_images(args):
    with h5py.File(args.path, mode='r') as h5fr:
        for task_id in h5fr['data'].keys():
            for demo_id in h5fr[f'data/{task_id}'].keys():
                ims = h5fr[f"data/{task_id}/{demo_id}/observations/image"][()]
                out_dir = os.path.join(args.save_img_out_dir, task_id, demo_id)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                for i in range(ims.shape[0]):
                    im = Image.fromarray(ims[i])
                    fname = str(i).zfill(3)
                    out_fpath = os.path.join(out_dir, f"{fname}.png")
                    im.save(out_fpath)
                quit()


def replay_actions(args):
    env = init_env("frka_base")
    with h5py.File(args.path, mode='r') as h5fr:
        for task_id in h5fr['data'].keys():
            for demo_id in h5fr[f'data/{task_id}'].keys():
                actions = h5fr[f'data/{task_id}/{demo_id}/actions'][()]
                env.reset()
                print("actions", actions)
                for action in actions:
                    env.step(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to hdf5 file")
    parser.add_argument("--show-tree", action="store_true", default=False)
    parser.add_argument("--save-img-out-dir", type=str, default=None)
    parser.add_argument("--film-strip-out-dir", type=str, default=None)
    args = parser.parse_args()

    np.set_printoptions(threshold=np.inf)

    if args.show_tree:
        f = nx.nxload(args.path)
        print(f.tree)

    with h5py.File(args.path, mode="r") as h5fr:
        import ipdb; ipdb.set_trace()
        for obj in h5fr.keys():
            pass

    np.set_printoptions(threshold=sys.maxsize)

    if args.save_img_out_dir is not None:
        if not os.path.exists(args.save_img_out_dir):
            os.makedirs(args.save_img_out_dir)
        save_images(args)
    else:
        save_film_strips(args)

    # replay_actions(args)
