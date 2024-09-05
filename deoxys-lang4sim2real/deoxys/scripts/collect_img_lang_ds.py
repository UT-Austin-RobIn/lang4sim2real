import argparse
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from deoxys.utils.params import *
from rlkit.lang4sim2real_utils.lang_templates import pp_lang_by_stages_v1


def create_z_mask_by_ts(actions_z):
    z_mask_by_ts = []
    z_key_ts = []
    for ts, action_z in enumerate(actions_z):
        if abs(action_z) < 0.15:
            mask_val = 0
        else:
            mask_val = int(np.sign(action_z))

        # Check if mask_val changed from last one
        if len(z_mask_by_ts) > 0 and mask_val != z_mask_by_ts[-1]:
            last_two = (z_mask_by_ts[-1], mask_val)
            if last_two == (0, -1):
                z_key_ts.append(ts)
            elif last_two == (-1, 0): 
                # This should be taken care of by the grasp ts.
                pass
            elif last_two == (0, 1):
                z_key_ts.append(ts)
            elif last_two == (1, 0):
                z_key_ts.append(ts)
            else:
                print("last_two", last_two)
                raise ValueError

        z_mask_by_ts.append(mask_val)
    return z_mask_by_ts, z_key_ts


def get_above_cont_with_obj_in_gripper_ts(ee_pos_xy, z_key_ts, success_ts):
    moving_over_cont_ts = np.arange(z_key_ts[-1], success_ts)
    for ts in moving_over_cont_ts:
        ee_pos_xy_over_cont = (
            (CONT_XY_LIMITS["lo"] <= ee_pos_xy[ts]).all()
            and (ee_pos_xy[ts] <= CONT_XY_LIMITS["hi"]).all())
        if ee_pos_xy_over_cont:
            return [ts]
    raise ValueError


def save_demos_img_lang(args, img_lang_dict_by_trajs):
    # based on robosuite-391r/robosuite/utils/data_collection_utils.py
    df_dict = dict(
        img_fname=[],
        lang=[],
    )

    for traj_idx, traj_dict in tqdm(enumerate(img_lang_dict_by_trajs)):
        imgs = traj_dict['images']
        langs = traj_dict['langs']
        assert len(imgs) == len(langs)
        for t in range(len(imgs)):
            # Save image to filename
            img = Image.fromarray(imgs[t])
            img_fname = (
                f"{args.obj_id}_{str(traj_idx).zfill(4)}_"
                f"{str(t).zfill(3)}.png")
            img.save(os.path.join(args.out_dir, img_fname))

            # Add entry to df dict
            df_dict['img_fname'].append(img_fname)
            df_dict['lang'].append(langs[t])

    # Save df as CSV
    df = pd.DataFrame(df_dict)
    out_csv_path = os.path.join(args.out_dir, "labels.csv")
    df.to_csv(out_csv_path)

    return out_csv_path


if __name__ == "__main__":
    """
    Given a buffer dataset, save images and lang captions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--buf-path", type=str, required=True)
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--task-id-to-keep", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(f"{args.out_dir}", exist_ok=True)

    obj_name = OBJECT_DETECTOR_CLASSES[args.obj_id]

    lang_by_stages = pp_lang_by_stages_v1(obj_name)

    img_lang_dict_by_trajs = []

    with h5py.File(args.buf_path, mode='r') as h5fr:
        assert str(args.task_id_to_keep) in h5fr['data'].keys()
        for task_id in [args.task_id_to_keep]:
            for demo_id in h5fr[f'data/{task_id}'].keys():
                rewards = h5fr[f'data/{task_id}/{demo_id}/rewards'][()]
                print("rewards", rewards)
                if rewards[-1] != 1:
                    continue

                actions = h5fr[f'data/{task_id}/{demo_id}/actions'][()]
                ee_pos_xy = (
                    h5fr[f'data/{task_id}/{demo_id}/observations/state'][
                        ()][:, :2])
                # ^ ee_pos is in [:3]

                T = actions.shape[0]
                actions_z = actions[:, 2]
                actions_gripper = actions[:, -1]
                print("actions_z", actions_z)
                print("actions_gripper", actions_gripper)
                gripper_open_list = [
                    int(bool(abs(1 + g_t)) < 1e-3) for g_t in actions_gripper]
                # ^ 1 for open, 0 for close

                try:
                    gripper_close_ts = gripper_open_list.index(0)
                    gripper_open_ts = (
                        gripper_open_list[gripper_close_ts:].index(1))
                    success_ts = gripper_close_ts + gripper_open_ts + 1
                    gripper_key_ts = [gripper_close_ts, success_ts]

                    z_mask_by_ts, z_key_ts = create_z_mask_by_ts(actions_z)
                    xy_key_ts = get_above_cont_with_obj_in_gripper_ts(
                        ee_pos_xy, z_key_ts, success_ts)
                    # print("z_mask_by_ts", z_mask_by_ts)

                    # print("gripper_key_ts", gripper_key_ts)
                    # print("z_key_ts", z_key_ts)
                except ValueError:
                    print("Unable to get gripper close/open ts.")
                    continue

                key_ts = (
                    sorted(list(set(gripper_key_ts + z_key_ts + xy_key_ts)))
                    + [T])
                print("key_ts", key_ts)
                if len(key_ts) != len(lang_by_stages):
                    print(f"Did not find {len(lang_by_stages)} key timesteps")
                    continue

                image_lang_dict = dict(
                    images=[],
                    langs=[],
                )
                traj_images = h5fr[
                    f'data/{task_id}/{demo_id}/observations/image'][()]

                stage = 0
                stages_by_ts = []
                for ts in range(T):
                    image = traj_images[ts]

                    # Find proper stage
                    if ts >= key_ts[stage]:
                        stage += 1

                    stages_by_ts.append(stage)
                    lang = lang_by_stages[stage]

                    image_lang_dict['images'].append(image)
                    image_lang_dict['langs'].append(lang)
                img_lang_dict_by_trajs.append(image_lang_dict)

    out_csv_path = save_demos_img_lang(args, img_lang_dict_by_trajs)
