import argparse
import collections
import os
import random
import time

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    get_timestamp_str)
from rlkit.torch.networks.resnet import ResNetMultiHead
import rlkit.util.experiment_script_utils as exp_utils
from rlkit.util.pythonplusplus import identity


class ImgGripperStateDataset(Dataset):
    def __init__(self, img_dir, hdf5_kwargs={}):
        self.hdf5_kwargs = hdf5_kwargs
        self.imgs, self.gripper_states = self.load_data_from_hdf5(img_dir)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        x = self.imgs[idx]
        y = self.gripper_states[idx]

        x = x.astype(np.float32).transpose(2, 0, 1) / 255.
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def load_data_from_hdf5(self, hdf5_path):
        imgs = []
        gripper_states = []
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
                    gripper_state = f[
                        f'data/{task_idx}/{demo_id}/observations/state'][
                        ()][:, -1]
                    gripper_state = gripper_state[()][:,None]  # (H, 1)
                    eepos = f[
                        f'data/{task_idx}/{demo_id}/observations/state'][
                        ()][:, :3]
                    gripper_states.append(
                        np.concatenate([eepos, gripper_state], axis=1))
                    task_idx_to_num_demos_loaded[task_idx] += 1

        imgs = np.concatenate(imgs, axis=0)
        # uint8 (dataset_size, img_h, img_w, 3)
        gripper_states = np.concatenate(gripper_states, axis=0)
        # (dataset_size, 4)
        return imgs, gripper_states


class GripperStateTrainer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dom1_num_demos_per_task = args.dom1_num_demos_per_task
        self.dom1_task_idxs = args.dom1_task_idxs
        self.ds1_train_loader, self.ds1_val_loader, self.ds1 = (
            self.get_train_val_loader(args.img_dir))
        self.model = self.init_model()
        self.optimizer = torch.optim.Adadelta(
            self.model.parameters(), lr=args.lr)
        self.val_epoch_losses = [np.inf]
        self.num_epochs = args.num_epochs

    def get_train_val_loader(self, img_dir):
        hdf5_kwargs_dom1 = {
            "task_indices": (
                exp_utils.create_task_indices_from_task_int_str_list(
                    self.dom1_task_idxs, np.inf)),
            "max_demos_per_task": self.dom1_num_demos_per_task,
        }

        ds = ImgGripperStateDataset(img_dir, hdf5_kwargs_dom1)

        train_subset_size = int(0.9 * len(ds))
        val_subset_size = len(ds) - train_subset_size
        ds_train_subset, ds_val_subset = torch.utils.data.random_split(
            ds, [train_subset_size, val_subset_size],
            generator=torch.Generator().manual_seed(1))
        ds_train_loader = torch.utils.data.DataLoader(
            ds_train_subset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, drop_last=False)
        ds_val_loader = torch.utils.data.DataLoader(
            ds_val_subset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, drop_last=False)
        return ds_train_loader, ds_val_loader, ds

    def init_model(self):
        cnn_params = dict()
        cnn_params.update(dict(
            fc_layers=[],
            conv_strides=[2, 2, 1, 1, 1],
            layers=[2, 2, 2, 2],
            num_channels=[16, 32, 64, 128],
            maxpool_stride=2,
            film_emb_dim_list=[],  # [386],
            num_film_inputs=0,  # 1,
            film_hidden_sizes=[],
            film_hidden_activation="identity",
            use_film_attn=False,
        ))
        self.device = torch.device('cuda')

        # predict 4-dimensional gripper state
        multihead_params = dict(
            fc_dims_by_head=[[3], [2]],
            # first head: predict (x, y, z) ee_pos.
            # second head: predict binary classif logits for ee open ([1, 0])
            # or ee closed ([0, 1])
            output_activations=[identity, nn.Softmax(dim=-1)],
        )
        cnn_params.update(multihead_params)
        # cnn_params['fc_layers'] = [4]

        # TODO: May need to process the targets
        # (some dimensions are a lot bigger than others)
        model = ResNetMultiHead(**cnn_params).to(self.device)
        return model

    def calc_loss(self, model_out_by_head, targets):
        loss_dict = {}
        loss_dict['ee_pos_pred'] = model_out_by_head[0]
        loss_dict['gripper_state_logits_pred'] = model_out_by_head[1]

        # Calculate gripper classification accuracy
        gripper_state_preds = torch.argmax(
            loss_dict['gripper_state_logits_pred'].detach(), dim=-1)
        gripper_state_targets = torch.tensor(
            targets[:, -1] < 0.07, dtype=torch.int64)
        # Remember that 0 (aka [1, 0]) == open, 1 (aka [0, 1]) == closed.
        loss_dict['gripper_classif_acc'] = torch.mean(
            (gripper_state_preds == gripper_state_targets).float()
        ).detach().item()

        # Calculate gripper classification f1
        # true positives, false positives, true negatives, false negatives
        tp = torch.sum(gripper_state_preds & gripper_state_targets)
        fp = torch.sum(gripper_state_preds & (1 - gripper_state_targets))
        tn = torch.sum((1 - gripper_state_preds) & (1 - gripper_state_targets))
        fn = torch.sum((1 - gripper_state_preds) & (gripper_state_targets))
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (
                2 * (precision * recall) / (precision + recall)
            ).detach().item()
        else:
            f1 = 0.0
        loss_dict['gripper_classif_f1'] = f1

        ee_pos_targets = targets[:, :3]
        # ee_pos_loss = F.mse_loss(loss_dict['ee_pos_pred'], ee_pos_targets)
        ee_pos_loss = F.l1_loss(loss_dict['ee_pos_pred'], ee_pos_targets)
        gripper_state_loss = F.cross_entropy(
            loss_dict['gripper_state_logits_pred'], gripper_state_targets)

        loss_dict["loss"] = 1.0 * ee_pos_loss + 10.0 * gripper_state_loss
        loss_dict["ee_pos_loss"] = ee_pos_loss
        loss_dict["gripper_state_loss"] = gripper_state_loss
        return loss_dict

    def run_train_or_val_epoch(self, ds1_loader, train=True):
        def run_model_forward(x, targets):
            preds = self.model(x)
            loss_dict = self.calc_loss(preds, targets)
            return loss_dict

        metric_dict = {
            "epoch_loss": 0.,
            "ee_pos_loss": 0.,
            "gripper_state_loss": 0.,
            "num_pts": len(ds1_loader.dataset),
            "ee_pos_l1_err": 0.,
            # "gripper_state_abs_err": 0.,
            "gripper_classif_acc": 0.,
            "gripper_f1": 0.,
        }
        if train:
            self.model.train()
        else:
            self.model.eval()
        for batch_idx, (img1, gs1) in enumerate(
                ds1_loader):
            img1 = img1.to(self.device).float()
            targets = gs1.to(self.device).float()

            if train:
                self.optimizer.zero_grad()
                loss_dict = run_model_forward(img1, targets)
            else:
                with torch.no_grad():
                    loss_dict = run_model_forward(img1, targets)
            metric_weight = (img1.shape[0] / metric_dict['num_pts'])

            # Loss metrics
            metric_dict['epoch_loss'] += (
                metric_weight * loss_dict["loss"].data.item())
            metric_dict['ee_pos_loss'] += (
                metric_weight * loss_dict["ee_pos_loss"].data.item())
            metric_dict['gripper_state_loss'] += (
                metric_weight * loss_dict["gripper_state_loss"].data.item())

            # Perf metrics
            metric_dict['ee_pos_l1_err'] += (
                metric_weight * torch.norm(
                    loss_dict['ee_pos_pred'] - targets[:, :3], p=1, dim=1
                ).mean().item())
            metric_dict['gripper_classif_acc'] += (
                metric_weight * loss_dict['gripper_classif_acc'])
            metric_dict['gripper_f1'] += (
                metric_weight * loss_dict['gripper_classif_f1'])
            if train:
                loss_dict["loss"].backward()
                self.optimizer.step()
        return metric_dict

    def get_log_str(self, metric_dict):
        log_str_items = []
        for key in metric_dict:
            if key not in ["num_pts"]:
                log_str_items.append(f"{key}: {round(metric_dict[key], 5)}")
        log_str = "\t".join(log_str_items)
        return log_str

    def train(self):
        for epoch_num in range(self.num_epochs):
            st = time.time()
            train_metric_dict = self.run_train_or_val_epoch(
                self.ds1_train_loader, train=True)
            val_metric_dict = self.run_train_or_val_epoch(
                self.ds1_val_loader, train=False)

            val_epoch_loss = val_metric_dict['epoch_loss']
            if val_epoch_loss < min(self.val_epoch_losses):
                model_out_path = os.path.join(args.out_dir, "best.pt")
                torch.save(self.model, model_out_path)
                print(f"saved best model to {model_out_path}")
            self.val_epoch_losses.append(val_epoch_loss)

            print(
                f"Epoch {epoch_num}" +
                "\tTrain: " + self.get_log_str(train_metric_dict) + "\n" +
                "\tVal: " + self.get_log_str(val_metric_dict) + "\n" +
                f"\tepoch time: {round(time.time() - st, 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num-epochs", required=True, type=int)
    parser.add_argument("--dom1-num-demos-per-task", type=int)
    parser.add_argument("--dom1-task-idxs", type=str, nargs="+", default=[])
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, get_timestamp_str())
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    trainer = GripperStateTrainer(args)
    trainer.train()
