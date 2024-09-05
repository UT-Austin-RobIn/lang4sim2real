import collections

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt


class TrajLabellerDataset(Dataset):
    def __init__(self, img_dir, hdf5_kwargs={}):
        self.hdf5_kwargs = hdf5_kwargs
        self.imgs, self.gripper_states, self.stage_nums, self.task_demo_ids = (
            self.load_data_from_hdf5(img_dir))

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        x = self.imgs[idx]
        gs = self.gripper_states[idx]
        sn = self.stage_nums[idx]
        task_demo_ids = self.task_demo_ids[idx]

        torch_x = x.astype(np.float32).transpose(0, 3, 1, 2) / 255.
        torch_x = torch.tensor(torch_x)
        gs = torch.tensor(gs)
        return torch_x, x, gs, sn, task_demo_ids

    def load_data_from_hdf5(self, hdf5_path):
        imgs = []
        gripper_states = []
        stage_nums = []
        task_demo_ids = []
        with h5py.File(hdf5_path, 'r', swmr=True, libver='latest') as f:
            max_demos_per_task = self.hdf5_kwargs.get(
                'max_demos_per_task', np.inf)
            task_idx_to_num_demos_loaded = collections.Counter()
            ds_task_idxs = [int(i) for i in list(f['data'].keys())]
            task_idxs_to_load = self.hdf5_kwargs.get(
                'task_indices', ds_task_idxs)
            for task_idx in task_idxs_to_load:
                demo_ids = list(f[f'data/{task_idx}'].keys())
                # if self.shuffle_demos:
                #     random.shuffle(demo_ids)
                for demo_id in tqdm(demo_ids):
                    # Break out if enough demos have been added.
                    if (task_idx_to_num_demos_loaded[task_idx]
                            >= max_demos_per_task):
                        print(
                            f"task_idx {task_idx}, loaded demo ids:",
                            demo_ids[:max_demos_per_task])
                        break
                    imgs.append(
                        f[f'data/{task_idx}/{demo_id}/observations/image'][()])
                    gripper_state = (
                        f[f'data/{task_idx}/{demo_id}/observations/state'][()][
                            :, -1])
                    gripper_state = gripper_state[()][:,None]  # (H, 1)
                    eepos = f[f'data/{task_idx}/{demo_id}/observations/state'][
                        ()][:, :3]  # (H, 3)
                    gripper_states.append(
                        np.concatenate([eepos, gripper_state], axis=1))
                    stage_num = f[
                        f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                    stage_nums.append(stage_num)
                    task_demo_ids.append(f'{task_idx}/{demo_id}')
                    task_idx_to_num_demos_loaded[task_idx] += 1

        imgs = np.array(imgs)  # imgs: uint8 (num_trajs, T, img_h, img_w, 3)
        gripper_states = np.array(gripper_states)  # gripper_states: (num_trajs, T, 4)
        stage_nums = np.array(stage_nums)  # stage_nums: (num_trajs, T)
        return imgs, gripper_states, stage_nums, task_demo_ids
 
    def write_predicted_stage_nums(self, hdf5_path, pred_stage_nums_dict):
        with h5py.File(hdf5_path, 'a', swmr=True, libver='latest') as f:
            ds_task_idxs = [int(i) for i in list(f['data'].keys())]
            task_idxs_to_load = self.hdf5_kwargs.get(
                'task_indices', ds_task_idxs)
            for task_idx in task_idxs_to_load:
                demo_ids = list(f[f'data/{task_idx}'].keys())
                for demo_id in tqdm(demo_ids):                    
                    stage_num = f[
                        f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                    if f'{task_idx}/{demo_id}' in pred_stage_nums_dict:
                        print(f"Writing {task_idx}/{demo_id} to hdf5")
                        if 'pred_stage_num' in f[f'data/{task_idx}/{demo_id}']:
                            del f[f'data/{task_idx}/{demo_id}/pred_stage_num']
                        f[f'data/{task_idx}/{demo_id}'].create_dataset(
                            'pred_stage_num', shape=stage_num.shape,
                            dtype='int32')
                        f[f'data/{task_idx}/{demo_id}/pred_stage_num'][:] = (
                            np.array(
                                pred_stage_nums_dict[f'{task_idx}/{demo_id}']))


def get_imgs_rewards_from_hdf5(path, max_imgs=0, task_index=0):
    with h5py.File(path, 'r') as f:
        f = f['data']
        imgs = []
        rewards = []
        stage = []
        for task_idx in f.keys():
            for demo_num in f[task_idx].keys():
                if len(rewards) >= max_imgs:
                    return (
                        imgs[:max_imgs], rewards[:max_imgs], stage[:max_imgs])
                demo = f[task_idx][demo_num]
                ar = demo['observations/image'][()]
                # num, x, y, c = ar.shape
                # ar = ar.reshape((num, c, x, y))
                imgs.append(ar)
                rewards.extend(demo['rewards'][:])
                stage.extend(demo['lang_stage_num'][:])
    return imgs, rewards, stage


def save_confusion_matrix(correct_stages, pred_stages):
    conf_matrix = sklearn.metrics.confusion_matrix(correct_stages, pred_stages)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save the figure as an image


if __name__ == "__main__":
    path = "/home/addie/object_detector_labeling/combined.hdf5"
    imgs, rewards = get_imgs_rewards_from_hdf5(path)
    print(imgs)
    print(rewards)
    print(len(imgs), "images of shape", imgs[0].shape)
    print(len(rewards), "rewards")
