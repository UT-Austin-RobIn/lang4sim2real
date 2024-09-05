import argparse
from datetime import datetime
import itertools
import os
import time

from colour import Color
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision

from rlkit.lang4sim2real_utils.lang_templates import ww_lang_by_stages_debug, ww_lang_by_stages_simplified
from rlkit.lang4sim2real_utils.train.img_lang_ds import (
    ImgLangDataset, ImgLangTimeContrDataset)
from rlkit.torch.networks.beta_vae import BetaVAE
from rlkit.torch.networks.image_augmentations import create_aug_transform_fns
from rlkit.torch.networks.resnet import ResNet, SpatialSoftmax
import rlkit.util.experiment_script_utils as exp_utils
import rlkit.util.pytorch_util as ptu
from rlkit.util.visualization_utils import plot_tsne_embs_by_stage

spatial_softmax = None

def get_timestamp_str():
    x = datetime.now()
    timestamp_str = (
        f"{x.year}-{str(x.month).zfill(2)}-{str(x.day).zfill(2)}"
        f"_{str(x.hour).zfill(2)}-{str(x.minute).zfill(2)}"
        f"-{str(x.second).zfill(2)}")
    print("timestamp_str", timestamp_str)
    return timestamp_str


def pairwise_diffs_matrix(X, Y, dist_fn):
    """returns matrix of shape (X.shape[0], X.shape[0], X.shape[1])
    where diffs[i][j] = X[i] - Y[j]"""
    assert len(X.shape) == 2
    if dist_fn == "l2":
        assert X.shape == Y.shape
        m, n = X.shape
        # we want to MINimize these for similar obs
        diffs = torch.norm(
            torch.reshape(X, (m, 1, n)) - Y, dim=-1)
    elif dist_fn == "dotprod":
        # we want to MAXimize these for similar obs
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
        diffs = X @ Y.T
    return diffs


def get_train_val_loader(
        img_dir, l2_unit_normalize, batch_size, drop_last=False,
        hdf5_kwargs={}, dataset_type="img-lang", two_stage_pp=0,
        override_lang_params={}, domain=1, rephrasing_csv="",
        env_name="", realrobot_target_obj="", out_dir="",
        mean_rephrase_emb=False, shuffle_demos=False,
        use_pred_stages=False):
    if dataset_type in ["img-lang", "img-recon"]:
        ds = ImgLangDataset(
            img_dir, l2_unit_normalize, hdf5_kwargs, two_stage_pp=two_stage_pp,
            override_lang_params=override_lang_params, domain=domain,
            rephrasing_csv=rephrasing_csv, env_name=env_name,
            realrobot_target_obj=realrobot_target_obj, out_dir=out_dir,
            mean_rephrase_emb=mean_rephrase_emb, shuffle_demos=shuffle_demos,
            use_pred_stages=use_pred_stages)
    elif dataset_type == "img-lang-time-contr":
        ds = ImgLangTimeContrDataset(
            img_dir, l2_unit_normalize, hdf5_kwargs)
    else:
        raise NotImplementedError

    train_subset_size = int(0.9 * len(ds))
    val_subset_size = len(ds) - train_subset_size
    ds_train_subset, ds_val_subset = torch.utils.data.random_split(
        ds, [train_subset_size, val_subset_size],
        generator=torch.Generator().manual_seed(1))
    ds_train_loader = torch.utils.data.DataLoader(
        ds_train_subset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=drop_last)
    ds_val_loader = torch.utils.data.DataLoader(
        ds_val_subset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=drop_last)
    print("train loader dataset size:", len(ds_train_loader.dataset))
    print("val loader dataset size:", len(ds_val_loader.dataset))
    return ds_train_loader, ds_val_loader, ds


def plot_traj(f_cnn, loss_type, hdf5_path, out_dir, device):
    f_cnn.eval()
    traj_idx = 0
    task_idx = 0
    with h5py.File(hdf5_path, 'r', swmr=True, libver='latest') as f:
        traj_imgs = f[f'data/{task_idx}/demo_{traj_idx}/observations/image'][()]
        traj_lang_idxs = f[f'data/{task_idx}/demo_{traj_idx}/lang_stage_num'][()]

    traj_imgs = ptu.from_numpy(traj_imgs).permute(0, 3, 1, 2).float() / 255.
    traj_imgs = traj_imgs.to(device)
    if loss_type in ["img-lang", "img-lang-time-contr"]:
        traj_embs = f_cnn(traj_imgs).cpu().detach().numpy()
    elif loss_type == "img-recon":
        # For beta-VAE:
        _, _, mu, log_var = f_cnn(traj_imgs)
        z = f_cnn.reparameterize(mu, log_var)
        traj_embs = z.cpu().detach().numpy()
    return plot_tsne_embs_by_stage(traj_embs, traj_lang_idxs, out_dir, "task0")


def calc_over_thresh_f1(preds, targets, thresh=0.5):
    preds_over_thresh = torch.where(preds > thresh, 1, 0)
    targets_over_thresh = torch.where(targets > thresh, 1, 0)
    both_preds_targets_over_thresh = preds_over_thresh * targets_over_thresh
    num_true_positives = torch.sum(both_preds_targets_over_thresh)
    num_positives = torch.sum(targets_over_thresh)
    precision = num_true_positives / (torch.sum(preds_over_thresh) + 1e-8)
    recall = num_true_positives / num_positives
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return f1.data.item()


def set_r3m_train_mode(model, unfrozen_mods):
    cnn = model.r3m.module.convnet
    if args.unfrozen_mods in ["adapters", "cnnlastlayer"]:
        cnn.eval()
        # first set everything to eval mode, then selectively set certain mods to training mode.
    else:
        return
    for i, layer in enumerate([cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4]):
        if args.unfrozen_mods == "adapters":
            for basic_block in layer:
                basic_block.conv_adapter.train()
        if args.unfrozen_mods == "cnnlastlayer" and i == 3:
            # Train current layer (layer == cnn.layer4)
            layer.train()


# Time contrastive similarity functions
def dotprod(X, Y):
    return torch.diag(X @ Y.T)


def negl2(X, Y):
    """
    X: (n, d)
    Y: (n, d)
    output: (n,)
    """
    return -torch.norm(X - Y, p=2, dim=-1)


def get_tc_sim_fn(sim_fn_type):
    if sim_fn_type == "dotprod":
        sim_fn = dotprod
    elif sim_fn_type == "negl2":
        sim_fn = negl2
    else:
        raise NotImplementedError
    return sim_fn
# END time contrastive similarity functions


def calc_time_contr_loss(out_t0, out_t1, out_t2, sim_fn_type):
    def get_row_shuffles(x, n):
        assert n >= 1
        shuffles = [x[torch.randperm(x.size()[0])] for _ in range(n)]
        return shuffles

    def get_exp_sim_with_negs(out, out_negs):
        assert isinstance(out_negs, list)
        exp_sim_w_negs = torch.cat([
            torch.exp(torch.diag(out @ out_neg.T))[None] for out_neg in out_negs], dim=0)
        exp_sim_w_negs = torch.sum(exp_sim_w_negs, dim=0)
        return exp_sim_w_negs

    sim_fn = get_tc_sim_fn(sim_fn_type)
    sim_0_1 = sim_fn(out_t0, out_t1)
    sim_0_2 = sim_fn(out_t0, out_t2)
    sim_1_2 = sim_fn(out_t1, out_t2)
    num_other_video_negs = 1
    out_t0_negs = get_row_shuffles(out_t0, num_other_video_negs)
    out_t2_negs = get_row_shuffles(out_t2, num_other_video_negs)
    exp_sim_t0_negs = get_exp_sim_with_negs(out_t0, out_t0_negs)
    exp_sim_t2_negs = get_exp_sim_with_negs(out_t2, out_t2_negs)
    eps = 1e-8
    loss_0 = -torch.log(
        eps + (
            torch.exp(sim_0_1) /
            (eps + torch.exp(sim_0_1) + torch.exp(sim_0_2) + exp_sim_t0_negs)
        )
    )
    loss_2 = -torch.log(
        eps + (
            torch.exp(sim_1_2) /
            (eps + torch.exp(sim_1_2) + torch.exp(sim_0_2) + exp_sim_t2_negs)
        )
    )
    loss = 0.5 * (torch.mean(loss_0) + torch.mean(loss_2))
    return loss


def calc_time_contr_loss_both_doms(out1, out2, loss, tc_alpha, metric_dict, sim_fn):
    bsz = out1.shape[0] // 3
    assert bsz == out2.shape[0] // 3

    # normalize
    out1 = F.normalize(out1, dim=-1)
    out2 = F.normalize(out2, dim=-1)

    out1_t0 = out1[:bsz]
    out1_t1 = out1[bsz:2*bsz]
    out1_t2 = out1[2*bsz:]
    out2_t0 = out2[:bsz]
    out2_t1 = out2[bsz:2*bsz]
    out2_t2 = out2[2*bsz:]
    loss_dom1 = calc_time_contr_loss(out1_t0, out1_t1, out1_t2, sim_fn)
    loss_dom2 = calc_time_contr_loss(out2_t0, out2_t1, out2_t2, sim_fn)
    time_contr_loss = 0.5 * (loss_dom1 + loss_dom2)
    metric_dict['epoch_time_contr_loss'] += (
        (bsz / metric_dict['num_pts']) * time_contr_loss.data.item())
    # print("loss before time_contr", loss)
    loss = ((1 - args.tc_alpha) * loss) + (args.tc_alpha * time_contr_loss)

    # time contr accuracy: random is 50%.
    metric_dict = calc_time_contr_acc(out1_t0, out1_t1, out1_t2, metric_dict, 1, sim_fn)
    metric_dict = calc_time_contr_acc(out2_t0, out2_t1, out2_t2, metric_dict, 2, sim_fn)
    return loss, metric_dict


def calc_time_contr_acc(out_t0, out_t1, out_t2, metric_dict, dom_i, sim_fn_type):
    bsz = out_t0.shape[0]
    sim_fn = get_tc_sim_fn(sim_fn_type)
    sim_0_1 = sim_fn(out_t0, out_t1)
    sim_0_2 = sim_fn(out_t0, out_t2)
    sim_1_2 = sim_fn(out_t1, out_t2)
    binary_acc_0 = torch.mean((sim_0_1 > sim_0_2).float())
    binary_acc_2 = torch.mean((sim_1_2 > sim_0_2).float())
    binary_acc = 0.5 * (binary_acc_0 + binary_acc_2)

    # We want this ratio to be as close to 0.0 as possible.
    if sim_fn_type == "dotprod":
        neg_pos_dotprod_ratio_0 = torch.mean(sim_0_2 / (1e-8 + sim_0_1))
        neg_pos_dotprod_ratio_2 = torch.mean(sim_0_2 / (1e-8 + sim_1_2))
    elif sim_fn_type == "negl2":
        neg_pos_dotprod_ratio_0 = torch.mean(sim_0_1 / (1e-8 + sim_0_2))
        neg_pos_dotprod_ratio_2 = torch.mean(sim_1_2 / (1e-8 + sim_0_2))
    else:
        raise NotImplementedError

    neg_pos_dotprod_ratio = 0.5 * (
        neg_pos_dotprod_ratio_0 + neg_pos_dotprod_ratio_2)
    metric_dict[f'time_contr_acc_dom{dom_i}'] += (
        (bsz / metric_dict['num_pts']) * binary_acc.data.item())
    metric_dict[f'time_contr_neg/pos_dotprod_dom{dom_i}'] += (
        (bsz / metric_dict['num_pts']) * neg_pos_dotprod_ratio.data.item())
    return metric_dict


def init_metric_dict(ds1_loader, ds2_loader, loss_type, variant):
    metric_dict = {
        "epoch_loss": 0.,
        "num_pts": min(
            len(ds1_loader.dataset), len(ds2_loader.dataset)),
    }
    if loss_type == "img-lang-time-contr":
        metric_dict.update({
            "epoch_time_contr_loss": 0.,
            "time_contr_acc_dom1": 0.,
            "time_contr_acc_dom2": 0.,
            "time_contr_neg/pos_dotprod_dom1": 0.,
            "time_contr_neg/pos_dotprod_dom2": 0.,
        })
    elif loss_type == "img-recon":
        metric_dict.update({
            "recons_loss": 0.,
            "kld": 0.,
            "C": 0.,
        })
    elif loss_type == "img-lang" and variant == "lang-dist-dotprod":
        metric_dict.update({
            "over_0.5_f1": 0.,
            "over_0.8_f1": 0.,
        })
    return metric_dict


def run_train_epoch(
        model, optimizer, ds1_train_loader, ds2_train_loader,
        xdomain_lang_embs_diff_mat, variant, device, img_aug_fns,
        output_stage, loss_type, tc_alpha, tc_sim_fn,
        lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, num_stages_per_task,
        args):
    global spatial_softmax
    metric_dict = init_metric_dict(
        ds1_train_loader, ds2_train_loader, loss_type, variant)
    model.train()
    if args.r3m:
        set_r3m_train_mode(model, args.unfrozen_mods)  # Freeze R3M net
    # train accuracy metric
    correct, total = 0, 0
    for batch_idx, ((img1, lang1, lang_idx1), (img2, lang2, lang_idx2)) in enumerate(
            zip(ds1_train_loader, ds2_train_loader)):
        img1 = img1.to(device).float()
        lang1 = lang1.to(device).float()
        lang_idx1 = lang_idx1.to(device).long()
        img2 = img2.to(device).float()
        lang2 = lang2.to(device).float()
        lang_idx2 = lang_idx2.to(device).long()

        # print(f"{img1.shape}, {img2.shape}")
        bsz = img1.shape[0]
        if img1.shape[0] != img2.shape[0]:
            bsz = min(img1.shape[0], img2.shape[0])
            img1 = img1[:bsz]
            img2 = img2[:bsz]
            lang1 = lang1[:bsz]
            lang2 = lang2[:bsz]
            lang_idx1 = lang_idx1[:bsz]
            lang_idx2 = lang_idx2[:bsz]

        # from PIL import Image
        # imarr = img1[::50]
        # imarr = np.uint8(255 * imarr.permute(0, 2, 3, 1).cpu())
        # imarr = np.concatenate(list(imarr), axis=0)
        # im = Image.fromarray(imarr)
        # im.save("before.png")

        if loss_type in ["img-lang", "img-recon"]:
            for aug_transform_fn in img_aug_fns:
                img1 = aug_transform_fn(img1)
                img2 = aug_transform_fn(img2)
        elif loss_type == "img-lang-time-contr":
            for i in range(img1.shape[1]):
                for aug_transform_fn in img_aug_fns:
                    img1[:, i] = aug_transform_fn(img1[:, i])
                    img2[:, i] = aug_transform_fn(img2[:, i])
            img1 = torch.cat([img1[:, i] for i in range(img1.shape[1])], dim=0)
            img2 = torch.cat([img2[:, i] for i in range(img2.shape[1])], dim=0)
            lang1 = torch.cat([lang1[:, i] for i in range(lang1.shape[1])], dim=0)
            lang2 = torch.cat([lang2[:, i] for i in range(lang2.shape[1])], dim=0)
            lang_idx1 = torch.cat([lang_idx1[:, i] for i in range(lang_idx1.shape[1])], dim=0)
            lang_idx2 = torch.cat([lang_idx2[:, i] for i in range(lang_idx2.shape[1])], dim=0)
        else:
            raise NotImplementedError

        # imarr = img1[:]# [::50]
        # imarr = np.uint8(255 * imarr.permute(0, 1, 3, 4, 2).cpu())
        # imarr_new = np.uint8(np.zeros((imarr.shape[0] * 128, imarr.shape[1] * 128, 3)))
        # for i in range(imarr.shape[1]):
        #     imarr_new[:, (128 * i):(128 * (i + 1))]  = np.concatenate(imarr[:, i], axis=0)
        # imarr = imarr_new
        # # imarr = np.uint8(255 * imarr.permute(0, 2, 3, 1).cpu())
        # # imarr = np.concatenate(list(imarr), axis=0)
        # im = Image.fromarray(imarr)
        # im.save("after.png")
        # quit()

        img = torch.cat([img1, img2], dim=0)

        optimizer.zero_grad()
        if args.r3m or args.loss_type == "img-recon":
            out = model(img)
        else:
            out = model(img, output_stage=output_stage)

        if args.spatial_softmax:
            if (spatial_softmax is None):
                print("initializing spatial softmax")
                spatial_softmax = SpatialSoftmax(
                    out.shape[2], out.shape[3], out.shape[1]).to(device)
            out = spatial_softmax(out)

        # Calculate loss
        if loss_type == "img-recon":
            loss_dict = model.loss_function(
                *out,
                M_N=0.00025)  # kld_weight
            loss = loss_dict['loss']
            metric_dict["recons_loss"] += (bsz / metric_dict['num_pts']) * (
                loss_dict['Reconstruction_Loss'].item())
            metric_dict["kld"] += (bsz / metric_dict['num_pts']) * (
                loss_dict['KLD'].item())
            metric_dict["C"] += (bsz / metric_dict['num_pts']) * (
                loss_dict['C'].item())
        elif loss_type == "img-lang-time-contr":
            out_dom_bsz = 3 * bsz
            out1 = out[:out_dom_bsz]
            out2 = out[out_dom_bsz:]
        else:
            out_dom_bsz = bsz
            out1 = out[:out_dom_bsz]
            out2 = out[out_dom_bsz:]

        if loss_type == "img-recon":
            pass
        elif variant == "stage-classif":
            loss_dom1 = F.cross_entropy(out1, lang_idx1 % num_stages_per_task)
            loss_dom2 = F.cross_entropy(out2, lang_idx2 % num_stages_per_task)
            loss = loss_dom1 + loss_dom2
        elif variant == "stage-reg":
            targets1 = ((lang_idx1 % num_stages_per_task) / num_stages_per_task)[:,None]
            targets2 = ((lang_idx2 % num_stages_per_task) / num_stages_per_task)[:,None]
            loss_dom1 = F.mse_loss(args.loss_arg_mult * out1, args.loss_arg_mult * targets1)
            loss_dom2 = F.mse_loss(args.loss_arg_mult * out2, args.loss_arg_mult * targets2)
            loss = loss_dom1 + loss_dom2
        elif variant == "lang-reg":
            lang1 = F.normalize(lang1, dim=-1)
            lang2 = F.normalize(lang2, dim=-1)
            out1 = F.normalize(out1, dim=-1)
            out2 = F.normalize(out2, dim=-1)
            loss = torch.norm(out1 - lang1, dim=-1).mean() + torch.norm(out2 - lang2, dim=-1).mean()
            accuracy, _ = get_lang_accuracy(
                out1, lang_idx1, lang1_idx_to_emb_mat,
                args.mean_rephrase_emb)
            correct += accuracy * lang_idx1.shape[0]
            total += lang_idx1.shape[0]
            accuracy, _ = get_lang_accuracy(
                out2, lang_idx2, lang2_idx_to_emb_mat,
                args.mean_rephrase_emb)
            correct += accuracy * lang_idx2.shape[0]
            total += lang_idx2.shape[0]
            metric_dict['accuracy'] = correct/total
        elif variant in ["lang-dist-l2", "lang-dist-dotprod"]:
            # Calculate loss
            dist_fn = variant.replace("lang-dist-", "")
            preds = pairwise_diffs_matrix(out1, out2, dist_fn)
            if dist_fn == "l2":
                targets = pairwise_diffs_matrix(lang1, lang2, dist_fn)
            elif dist_fn == "dotprod":
                idxs = torch.tensor(list(itertools.product(lang_idx1, lang_idx2)))
                # gives us a list of tuples of the coordinates from which we want
                # to pluck the xdomain_lang_embs_diff_mat
                targets = xdomain_lang_embs_diff_mat[idxs[:,0], idxs[:,1]].reshape(
                    out_dom_bsz, out_dom_bsz)
                # These matrices are (bsz, bsz)
                metric_dict['over_0.5_f1'] += (bsz / metric_dict['num_pts']) * (
                    calc_over_thresh_f1(preds, targets, thresh=0.5))
                metric_dict['over_0.8_f1'] += (bsz / metric_dict['num_pts']) * (
                    calc_over_thresh_f1(preds, targets, thresh=0.8))
            # Img-lang loss
            loss = F.mse_loss(
                args.loss_arg_mult * preds, args.loss_arg_mult * targets)
        else:
            raise NotImplementedError

        if loss_type == "img-lang-time-contr":
            loss, metric_dict = calc_time_contr_loss_both_doms(
                out1, out2, loss, tc_alpha, metric_dict, tc_sim_fn)
        metric_dict['epoch_loss'] += (
            (bsz / metric_dict['num_pts']) * loss.data.item())

        loss.backward()
        optimizer.step()
    # print(f"epoch {epoch} train loss: {epoch_loss}")
    return metric_dict


def get_lang_accuracy(out, lang_idx, lang_idx_to_emb_mat, mean_rephrase_emb):
    correct = 0
    total = 0
    total += lang_idx.size(0)
    predicted_idx = np.zeros((lang_idx.size(0)))

    # if rephrase_lang = True, then lang_idx_to_emb_mat has shape
    # (num_tasks * num_stages/task, num_rephrasings + 1, 384)
    if mean_rephrase_emb and len(lang_idx_to_emb_mat.shape) == 3:
        num_rephrasings = 1
        rephrase_lang_idx_to_emb_mat = torch.mean(lang_idx_to_emb_mat, dim=1)
    elif len(lang_idx_to_emb_mat.shape) == 3:
        num_rephrasings = lang_idx_to_emb_mat.shape[1]
        rephrase_lang_idx_to_emb_mat = torch.cat(list(lang_idx_to_emb_mat), dim=0)
    else:
        num_rephrasings = 1
        rephrase_lang_idx_to_emb_mat = lang_idx_to_emb_mat

    for i in range(lang_idx.size(0)):
        distances = torch.norm(out[i] - rephrase_lang_idx_to_emb_mat, dim=1)
        # collapse predicted_idx to be correct if it hits any of the rephrasing indices.
        # this is also used when plotting the confusion matrix.
        predicted_idx[i] = torch.argmin(distances).item() // num_rephrasings
        correct += int(predicted_idx[i] == lang_idx[i])

    cm = get_cm(predicted_idx, lang_idx.cpu(), lang_idx_to_emb_mat.shape[0])
    return (correct / total), cm


def get_cm(predicted_idx, actual_idx, num_classes):
    labels = [i for i in range(num_classes)]
    cm = metrics.confusion_matrix(actual_idx, predicted_idx, labels=labels).astype(int)
    return cm


def run_val_epoch(
        model, ds1_val_loader, ds2_val_loader,
        xdomain_lang_embs_diff_mat, variant, device,
        output_stage, loss_type, tc_alpha, tc_sim_fn,
        lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, num_stages_per_task):
    global spatial_softmax
    metric_dict = init_metric_dict(
        ds1_val_loader, ds2_val_loader, loss_type, variant)
    model.eval()
    correct = 0
    total = 0
    if variant in ["stage-classif", "stage-reg"]:
        confusion_matrix_dom1 = np.zeros((num_stages_per_task, num_stages_per_task), dtype=int)
        confusion_matrix_dom2 = np.zeros((num_stages_per_task, num_stages_per_task), dtype=int)
    elif variant == "lang-reg":
        confusion_matrix_dom1 = np.zeros((lang1_idx_to_emb_mat.shape[0], lang1_idx_to_emb_mat.shape[0]), dtype=int)
        confusion_matrix_dom2 = np.zeros((lang2_idx_to_emb_mat.shape[0], lang2_idx_to_emb_mat.shape[0]), dtype=int)

    for ((img1, lang1, lang_idx1), (img2, lang2, lang_idx2)) in zip(
            ds1_val_loader, ds2_val_loader):
        img1 = img1.to(device).float()
        lang1 = lang1.to(device).float()
        lang_idx1 = lang_idx1.to(device).long()
        img2 = img2.to(device).float()
        lang2 = lang2.to(device).float()
        lang_idx2 = lang_idx2.to(device).long()

        bsz = img1.shape[0]
        if img1.shape[0] != img2.shape[0]:
            bsz = min(img1.shape[0], img2.shape[0])
            img1 = img1[:bsz]
            img2 = img2[:bsz]
            lang1 = lang1[:bsz]
            lang2 = lang2[:bsz]
            lang_idx1 = lang_idx1[:bsz]
            lang_idx2 = lang_idx2[:bsz]

        if loss_type == "img-lang-time-contr":
            img1 = torch.cat([img1[:, i] for i in range(img1.shape[1])], dim=0)
            img2 = torch.cat([img2[:, i] for i in range(img2.shape[1])], dim=0)
            lang1 = torch.cat([lang1[:, i] for i in range(lang1.shape[1])], dim=0)
            lang2 = torch.cat([lang2[:, i] for i in range(lang2.shape[1])], dim=0)
            lang_idx1 = torch.cat([lang_idx1[:, i] for i in range(lang_idx1.shape[1])], dim=0)
            lang_idx2 = torch.cat([lang_idx2[:, i] for i in range(lang_idx2.shape[1])], dim=0)

        img = torch.cat([img1, img2], dim=0)

        with torch.no_grad():
            if args.r3m or args.loss_type == "img-recon":
                out = model(img)
            else:
                out = model(img, output_stage=output_stage)
            if args.spatial_softmax:
                out = spatial_softmax(out)

            # Calculate loss
            if loss_type == "img-recon":
                loss_dict = model.loss_function(
                    *out,
                    M_N=0.00025)  # kld_weight
                loss = loss_dict['loss']
                # print("val loss_dict", loss_dict)
                metric_dict["recons_loss"] += (bsz / metric_dict['num_pts']) * (
                    loss_dict['Reconstruction_Loss'].item())
                metric_dict["kld"] += (bsz / metric_dict['num_pts']) * (
                    loss_dict['KLD'].item())
                metric_dict["C"] += (bsz / metric_dict['num_pts']) * (
                    loss_dict['C'].item())

                # Save reconstructed images
                recon = out[0]
                input_ = out[1]
                recon_to_save = recon[::10].permute(0, 2, 3, 1)  # (2*bsz / 10, img_size, img_size, 3)
                recon_to_save = torch.cat(list(recon_to_save), dim=1)  # (img_size, (2*bsz/10) * img_size, 3)
                input_to_save = input_[::10].permute(0, 2, 3, 1)  # (2*bsz / 10, img_size, img_size, 3)
                input_to_save = torch.cat(list(input_to_save), dim=1)  # (img_size, (2*bsz/10) * img_size, 3)
                exs_to_save = torch.cat([recon_to_save, input_to_save], dim=0)  # (2 * img_size, (2*bsz/10) * img_size, 3)
                exs_to_save = np.uint8(255 * ptu.get_numpy(exs_to_save))
                metric_dict['im_recons'] = Image.fromarray(exs_to_save)
            elif loss_type == "img-lang-time-contr":
                out_dom_bsz = 3 * bsz
                out1 = out[:out_dom_bsz]
                out2 = out[out_dom_bsz:]
            else:
                out_dom_bsz = bsz
                out1 = out[:out_dom_bsz]
                out2 = out[out_dom_bsz:]

            if loss_type == "img-recon":
                pass
            elif variant in ["stage-classif", "stage-reg"]:
                actual_stage1 = lang_idx1 % num_stages_per_task
                actual_stage2 = lang_idx2 % num_stages_per_task
                if variant == "stage-classif":
                    loss = (F.cross_entropy(out1, actual_stage1)
                        + F.cross_entropy(out2, actual_stage2))
                    predicted_stage1 = torch.argmax(out1, dim=1)
                    predicted_stage2 = torch.argmax(out2, dim=1)
                elif variant == "stage-reg":
                    targets1 = (actual_stage1 / num_stages_per_task)[:, None]
                    targets2 = (actual_stage2 / num_stages_per_task)[:, None]
                    loss_dom1 = F.mse_loss(args.loss_arg_mult * out1, args.loss_arg_mult * targets1)
                    loss_dom2 = F.mse_loss(args.loss_arg_mult * out2, args.loss_arg_mult * targets2)
                    loss = loss_dom1 + loss_dom2
                    predicted_stage1 = torch.round(out1 * num_stages_per_task)
                    predicted_stage2 = torch.round(out2 * num_stages_per_task)
                correct += (predicted_stage1 == actual_stage1).sum().item()
                correct += (predicted_stage2 == actual_stage2).sum().item()
                total += out1.shape[0] + out2.shape[0]
                confusion_matrix_dom1 += get_cm(predicted_stage1.cpu(), actual_stage1.cpu(), num_stages_per_task)
                confusion_matrix_dom2 += get_cm(predicted_stage2.cpu(), actual_stage2.cpu(), num_stages_per_task)
            elif variant == "lang-reg":
                lang1 = F.normalize(lang1, dim=-1)
                lang2 = F.normalize(lang2, dim=-1)
                out1 = F.normalize(out1, dim=-1)
                out2 = F.normalize(out2, dim=-1)
                loss = (torch.norm(out1 - lang1, dim=-1).mean()
                    + torch.norm(out2 - lang2, dim=-1).mean())
                accuracy, cm1 = get_lang_accuracy(
                    out1, lang_idx1, lang1_idx_to_emb_mat,
                    args.mean_rephrase_emb)
                correct += accuracy * lang_idx1.shape[0]
                total += lang_idx1.shape[0]
                confusion_matrix_dom1 += cm1
                accuracy, cm2 = get_lang_accuracy(
                    out2, lang_idx2, lang2_idx_to_emb_mat,
                    args.mean_rephrase_emb)
                correct += accuracy * lang_idx2.shape[0]
                total += lang_idx2.shape[0]
                confusion_matrix_dom2 += cm2
            elif variant in ["lang-dist-l2", "lang-dist-dotprod"]:
                dist_fn = variant.replace("lang-dist-", "")
                preds = pairwise_diffs_matrix(out1, out2, dist_fn)
                if dist_fn == "l2":
                    targets = pairwise_diffs_matrix(lang1, lang2, dist_fn)
                elif dist_fn == "dotprod":
                    # idxs = torch.tensor(list(itertools.product(lang_idx1, lang_idx2)))
                    idxs = torch.cartesian_prod(lang_idx1, lang_idx2)
                    # gives us a list of tuples of the coordinates from which we want
                    # to pluck the xdomain_lang_embs_diff_mat
                    targets = xdomain_lang_embs_diff_mat[idxs[:,0], idxs[:,1]].reshape(
                        out_dom_bsz, out_dom_bsz)
                    # These matrices are (bsz, bsz)
                    metric_dict['over_0.5_f1'] += (bsz / metric_dict['num_pts']) * (
                        calc_over_thresh_f1(preds, targets, thresh=0.5))
                    metric_dict['over_0.8_f1'] += (bsz / metric_dict['num_pts']) * (
                        calc_over_thresh_f1(preds, targets, thresh=0.8))
                loss = F.mse_loss(args.loss_arg_mult * preds, args.loss_arg_mult * targets)
            else:
                raise NotImplementedError
            if loss_type == "img-lang-time-contr":
                loss, metric_dict = calc_time_contr_loss_both_doms(
                    out1, out2, loss, tc_alpha, metric_dict, tc_sim_fn)
            metric_dict['epoch_loss'] += (
                (bsz / metric_dict['num_pts']) * loss.data.item())
    if total != 0:
        metric_dict['accuracy'] = correct/total
        metric_dict['confusion_matrix_dom1'] = confusion_matrix_dom1
        metric_dict['confusion_matrix_dom2'] = confusion_matrix_dom2
    # print(f"epoch {epoch} val loss: {epoch_loss}")
    return metric_dict


def compute_xdomain_lang_embs_diff_mat(
        lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, device,
        target_diff_mat_path=None, num_stages_per_task=0):
    if target_diff_mat_path is not None:
        diff_mat = torch.tensor(np.load(target_diff_mat_path)).float()
        return diff_mat.to(device)
    assert lang1_idx_to_emb_mat.shape[0] % num_stages_per_task == 0
    assert lang2_idx_to_emb_mat.shape[0] % num_stages_per_task == 0
    num_dom2_tasks = lang2_idx_to_emb_mat.shape[0] // num_stages_per_task
    diff_mat = lang1_idx_to_emb_mat.cpu() @ lang2_idx_to_emb_mat.cpu().T
    softmax_temp = 0.01
    diff_mats = []
    for dom2_tr_task_i in range(num_dom2_tasks):
        start_idx = dom2_tr_task_i * num_stages_per_task
        end_idx = (dom2_tr_task_i + 1) * num_stages_per_task
        diff_mat_i = softmax(diff_mat[:, start_idx:end_idx] / softmax_temp, axis=-1)
        diff_mats.append(diff_mat_i)
    diff_mat = torch.cat(diff_mats, axis=-1)
    return diff_mat.clone().detach().to(device)


def train(args):
    if args.override_lang:
        ds1_override_lang_params = {
            "fn": ww_lang_by_stages_debug,
            "fn_kwargs": dict(
                grasp_obj_name="last bead",
                flex_wraparound_obj_name="beads",
                central_obj_name="cylinder"),
        }
        ds2_override_lang_params = {
            "fn": ww_lang_by_stages_debug,
            "fn_kwargs": dict(
                grasp_obj_name="bridge",
                flex_wraparound_obj_name="ethernet cable",
                central_obj_name="3d printer spool"),
        }
        print("ds1_override_lang_params", ds1_override_lang_params)
        print("ds2_override_lang_params", ds2_override_lang_params)
        input("Overriding language stages found in buffer. continue?")
    else:
        ds1_override_lang_params = {}
        ds2_override_lang_params = {}

    l2_unit_normalize_target_embs = bool(args.variant != "lang-dist-l2")
    ds1_train_loader, ds1_val_loader, ds1 = get_train_val_loader(
        args.dom1_img_dir, l2_unit_normalize_target_embs, args.batch_size,
        hdf5_kwargs=args.hdf5_kwargs_dom1, dataset_type=args.loss_type,
        two_stage_pp=args.two_stage_pp, override_lang_params=ds1_override_lang_params,
        domain=1, rephrasing_csv=args.rephrasing_csv, env_name=args.env[0] if args.env else "",
        out_dir=args.out_dir, mean_rephrase_emb=args.mean_rephrase_emb,
        shuffle_demos=args.shuffle_demos)
    ds2_train_loader, ds2_val_loader, ds2 = get_train_val_loader(
        args.dom2_img_dir, l2_unit_normalize_target_embs, args.batch_size,
        hdf5_kwargs=args.hdf5_kwargs_dom2, dataset_type=args.loss_type,
        two_stage_pp=args.two_stage_pp, override_lang_params=ds2_override_lang_params,
        domain=2, rephrasing_csv=args.rephrasing_csv, env_name=args.env[1] if args.env else "",
        realrobot_target_obj=args.realrobot_target_obj, out_dir=args.out_dir,
        mean_rephrase_emb=args.mean_rephrase_emb,
        shuffle_demos=args.shuffle_demos, use_pred_stages=args.use_pred_stages)  # only dom2 is possibly a real env.
    lang1_idx_to_emb_mat = ds1.unique_lang_idx_to_lang_emb_matrix
    lang2_idx_to_emb_mat = ds2.unique_lang_idx_to_lang_emb_matrix
    task_idx_to_num_stages_map1 = ds1.task_idx_to_num_stages_map
    task_idx_to_num_stages_map2 = ds2.task_idx_to_num_stages_map
    num_stages_per_task = None
    all_tasks_num_stages1 = np.array(list(task_idx_to_num_stages_map1.values()))
    all_tasks_num_stages2 = np.array(list(task_idx_to_num_stages_map2.values()))
    num_stages_per_task = all_tasks_num_stages1[0]
    # make sure all task_idxs have the same number of stages for this baseline.
    assert ((num_stages_per_task == all_tasks_num_stages1).all()
        and (num_stages_per_task == all_tasks_num_stages2).all())
    print(f"We are assuming {num_stages_per_task} stages per task.")
    time.sleep(2)

    device = torch.device('cuda')

    if args.variant in ["lang-reg", "stage-classif", "stage-reg"]:
        xdomain_lang_embs_diff_mat = None
    elif args.variant in ["lang-dist-l2", "lang-dist-dotprod"]:
        xdomain_lang_embs_diff_mat = compute_xdomain_lang_embs_diff_mat(
            lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, device,
            args.target_diff_mat_path, num_stages_per_task)
    else:
        raise NotImplementedError
    print("diff_mat\n", xdomain_lang_embs_diff_mat)

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

    if args.variant == "stage-classif":
        output_stage = ""  # want prediction to go through fc layers
        cnn_params['fc_layers'] = [num_stages_per_task]
        cnn_params['output_activation'] = torch.nn.Softmax(dim=-1)
    elif args.variant == "stage-reg":
        output_stage = ""  # want prediction to go through fc layers
        cnn_params['fc_layers'] = [1]
    elif args.variant == "lang-reg":
        output_stage = ""  # wnat prediction to go through fc layers
        cnn_params['fc_layers'] = [384]
    elif args.spatial_softmax:
        output_stage = "conv_channels"
    else:
        output_stage = ""

    if args.r3m:
        from rlkit.torch.networks.cnn import R3MWrapper
        kwargs = {}
        if args.unfrozen_mods == "adapters":
            kwargs = dict(
                strict_param_load=False,
                adapter_kwargs=dict(
                    compress_ratio=args.adapter_compress_ratio),
            )
        model = R3MWrapper(device=device, **kwargs)
        if args.unfrozen_mods != "all":
            model.r3m.module.freeze()
        if args.unfrozen_mods == "cnnlastlayer":
            model.r3m.module.unfreeze_last_layer()
        if args.unfrozen_mods == "adapters":
            model.r3m.module.convnet.unfreeze_conv_adapters()
        assert args.variant == "lang-reg" or args.target_diff_mat_path
    elif args.img_net:
        from rlkit.torch.networks.cnn import ImgNetWrapper
        model = ImgNetWrapper(device=device)
    elif args.loss_type == "img-recon":
        model = BetaVAE(
            in_channels=3,
            latent_dim=128,
            beta=4,
            gamma=10.0,
            Capacity_max_iter=10000,
            loss_type='B',
        ).to(device)
    else:
        model = ResNet(**cnn_params).to(device)
    *_, c, h, w = next(iter(ds1_train_loader))[0].shape
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    print("args.img_aug", args.img_aug)
    transf_kwargs = {
            "image_size":  (h, w),
            "im_aug_pad": args.pad_size,
            "rnd_erase_prob": 0.0,
            "aug_transforms": args.img_aug, #["pad_crop", "bright_contr", "rgb_shift", "erase"],
        }
    img_aug_fns = create_aug_transform_fns(transf_kwargs) # do i need to check the images are the same size
    df = pd.DataFrame(columns=['Epoch', 'Train loss', 'Train f1>0.5', 'Val loss', 'Val f1>0.5', 'Val f1>0.8'])

    # Val loss before training
    # val_epoch_loss = run_val_epoch(
    #     model, ds1_val_loader, ds2_val_loader, device)
    # print(f"Val loss before training: {round(val_epoch_loss, 5)}")

    val_epoch_losses = [np.inf]
    for epoch in range(args.num_epochs):
        st = time.time()
        # TODO: Clean up -- pass in args instead of args.[attr]
        train_metric_dict = run_train_epoch(
            model, optimizer, ds1_train_loader, ds2_train_loader,
            xdomain_lang_embs_diff_mat, args.variant, device, img_aug_fns,
            output_stage, args.loss_type, args.tc_alpha, args.tc_sim_fn,
            lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, num_stages_per_task,
            args)
        train_epoch_loss = train_metric_dict['epoch_loss']
        csv_row = {
            'Epoch': epoch,
            'Train loss': round(train_epoch_loss, 5),
        }
        log_str = f"epoch {epoch}:  Train loss: {csv_row['Train loss']}"
        if 'over_0.5_f1' in train_metric_dict:
            csv_row['Train f1>0.5'] = round(train_metric_dict['over_0.5_f1'], 5)
            csv_row['Train f1>0.8'] = round(train_metric_dict['over_0.8_f1'], 5)
            log_str += f"  f1>0.5: {csv_row['Train f1>0.5']}"
            log_str += f"  f1>0.8: {csv_row['Train f1>0.8']}"
        if 'epoch_time_contr_loss' in train_metric_dict:
            csv_row['Train time contr loss'] = round(train_metric_dict['epoch_time_contr_loss'], 5)
            csv_row['Train time contr acc1'] = round(train_metric_dict['time_contr_acc_dom1'], 5)
            csv_row['Train time contr acc2'] = round(train_metric_dict['time_contr_acc_dom2'], 5)
            csv_row['Train time contr neg/pos ratio1'] = round(train_metric_dict['time_contr_neg/pos_dotprod_dom1'], 5)
            csv_row['Train time contr neg/pos ratio2'] = round(train_metric_dict['time_contr_neg/pos_dotprod_dom2'], 5)
            log_str += f"  Time Contr loss: {csv_row['Train time contr loss']}"
            log_str += ("\n\t\t Train time contr acc: " +
                f"({csv_row['Train time contr acc1']}, {csv_row['Train time contr acc2']})" +
                f"\tneg/pos ratio: ({csv_row['Train time contr neg/pos ratio1']}, {csv_row['Train time contr neg/pos ratio2']}) \n")
        if 'accuracy' in train_metric_dict:
            csv_row['Train accuracy'] = round(train_metric_dict['accuracy'], 5)
            log_str += f"  Train accuracy: {csv_row['Train accuracy']}\n"
        if 'recons_loss' in train_metric_dict:
            csv_row['Train recons_loss'] = round(train_metric_dict['recons_loss'], 5)
            csv_row['Train kld'] = round(train_metric_dict['kld'], 5)
            csv_row['C'] = round(train_metric_dict['C'], 5)
            log_str += (
                f" \tTrain recons loss: {csv_row['Train recons_loss']}"
                + f" \tTrain kld: {csv_row['Train kld']}")

        if epoch % args.val_freq == 0:
            val_metric_dict = run_val_epoch(
                model, ds1_val_loader, ds2_val_loader,
                xdomain_lang_embs_diff_mat, args.variant, device,
                output_stage, args.loss_type, args.tc_alpha, args.tc_sim_fn,
                lang1_idx_to_emb_mat, lang2_idx_to_emb_mat, num_stages_per_task)
            val_epoch_loss = val_metric_dict['epoch_loss']
            val_epoch_dict = {
                'Val loss': round(val_epoch_loss, 5),
            }

            if args.variant in ["lang-reg", "stage-classif", "stage-reg"] and args.loss_type == "img-lang":
                plt.close()
                plot_confusion_matrix(val_metric_dict["confusion_matrix_dom1"], args, epoch, dom_id=1)
                plot_confusion_matrix(val_metric_dict["confusion_matrix_dom2"], args, epoch, dom_id=2)

            log_str += f"\t || Val loss: {val_epoch_dict['Val loss']}"
            if 'over_0.5_f1' in val_metric_dict:
                val_epoch_dict['Val f1>0.5'] = round(val_metric_dict['over_0.5_f1'], 5)
                val_epoch_dict['Val f1>0.8'] = round(val_metric_dict['over_0.8_f1'], 5)
                log_str += f"  f1>0.5: {val_epoch_dict['Val f1>0.5']}"
                log_str += f"  f1>0.8: {val_epoch_dict['Val f1>0.8']}"
            if 'epoch_time_contr_loss' in val_metric_dict:
                val_epoch_dict['Val time contr loss'] = round(val_metric_dict['epoch_time_contr_loss'], 5)
                val_epoch_dict['Val time contr acc1'] = round(val_metric_dict['time_contr_acc_dom1'], 5)
                val_epoch_dict['Val time contr acc2'] = round(val_metric_dict['time_contr_acc_dom2'], 5)
                val_epoch_dict['Val time contr neg/pos ratio1'] = round(val_metric_dict['time_contr_neg/pos_dotprod_dom1'], 5)
                val_epoch_dict['Val time contr neg/pos ratio2'] = round(val_metric_dict['time_contr_neg/pos_dotprod_dom2'], 5)
                log_str += f"  Time Contr loss: {val_epoch_dict['Val time contr loss']}"
                log_str += ("\n\t\t Val time contr acc: " +
                    f"({val_epoch_dict['Val time contr acc1']}, {val_epoch_dict['Val time contr acc2']})" +
                    f"\tneg/pos ratio: ({val_epoch_dict['Val time contr neg/pos ratio1']}, {val_epoch_dict['Val time contr neg/pos ratio2']})")
            if 'accuracy' in val_metric_dict:
                val_epoch_dict['Val accuracy'] = round(val_metric_dict['accuracy'], 5)
                log_str += f"  Val accuracy: {val_epoch_dict['Val accuracy']}\n"
            if 'recons_loss' in train_metric_dict:
                csv_row['Val recons_loss'] = round(val_metric_dict['recons_loss'], 5)
                csv_row['Val kld'] = round(val_metric_dict['kld'], 5)
                csv_row['Val C'] = round(val_metric_dict['C'], 3)
                log_str += (
                    f" \tVal recons loss: {csv_row['Val recons_loss']}"
                    + f" \tVal kld: {csv_row['Val kld']}"
                    + f" \tVal C: {csv_row['Val C']}")

            if val_epoch_loss < min(val_epoch_losses):
                model_out_path = os.path.join(args.out_dir, "best.pt")
                torch.save(model, model_out_path)
                print(f"saved best model to {model_out_path}")
                if epoch % args.plot_freq == 0:
                    plot_traj(
                        model, args.loss_type, args.dom1_img_dir, args.out_dir,
                        device)
                if args.loss_type == "img-recon":
                    im_recons_path = os.path.join(
                        args.out_dir, "reconstructions.png")
                    print("im_recons_path", im_recons_path)
                    val_metric_dict['im_recons'].save(im_recons_path)
            if ((args.save_ckpt_freq is not None)
                    and ((epoch + 1) % args.save_ckpt_freq == 0)
                    and (epoch > 0)):
                # copy the best.pt to a filename with the epoch number
                model_epoch_path = os.path.join(
                    args.out_dir, f"best_before_epoch_{epoch + 1}.pt")
                command = f"cp {model_out_path} {model_epoch_path}"
                print(command)
                os.system(command)
            val_epoch_losses.append(val_epoch_loss)
            csv_row.update(val_epoch_dict)

        log_str += f" || Epoch Time: {round(time.time() - st, 1)}"
        print(log_str)
        df = df.append(csv_row, ignore_index=True)
    df.to_csv(os.path.join(args.out_dir, "data.csv"), index=False)


def plot_vals(path, save_path, y):
    assert len(y) >= 1
    df = pd.read_csv(path)
    epochs = df['Epoch']
    title = ""
    for str in y:
        y_vals = df[y]
        plt.plot(epochs, y_vals, marker='o', linestyle='-', color='b')
        title += str + ", "
    y_str = title[0: -2]
    plt.title(y_str + ' Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(y_str)
    plt.grid(True)
    save = os.path.join(save_path, 'plot' + y_str.replace(' ', '_') + '.png')
    plt.savefig(save)
    plt.close()


def plot_confusion_matrix(cm, args, epoch, dom_id):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[i for i in range(cm.shape[0])], yticklabels=[i for i in range(cm.shape[0])], annot_kws={"size": 5})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_dir = os.path.join(args.out_dir, f"confusion_matrices")
    if not os.path.exists(cm_dir):
        os.mkdir(cm_dir)
    plt.savefig(os.path.join(cm_dir, f"confusion_matrix_dom{dom_id}_epoch{epoch}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, help="Path to hdf5 file")
    parser.add_argument("--dom1-img-dir", type=str, required=True)
    parser.add_argument("--dom1-task-idxs", type=str, nargs="+", default=[])
    parser.add_argument("--dom1-num-demos-per-task", type=int, default=None)
    parser.add_argument("--dom2-img-dir", type=str, required=True)
    parser.add_argument("--dom2-task-idxs", type=str, nargs="+", default=[])
    parser.add_argument("--dom2-num-demos-per-task", type=int, default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--loss-arg-mult", type=float, default=None)
    parser.add_argument("--img-aug", type=str, nargs="+", default=[])
    parser.add_argument("--pad-size", type=int, default=4)
    parser.add_argument("--spatial-softmax", action="store_true", default=False)
    parser.add_argument("--target-diff-mat-path", type=str, default=None)
    parser.add_argument(
        "--loss-type", type=str, default="img-lang",
        choices=["img-lang", "img-lang-time-contr", "img-recon"])
    parser.add_argument("--tc-alpha", type=float, default=0.0)
    parser.add_argument("--tc-sim-fn", type=str, choices=["negl2", "dotprod"], default=None)
    parser.add_argument("--plot-freq", type=int, default=1)
    parser.add_argument(
        "--save-ckpt-freq", type=int, default=None,
        help="Save the best checkpoint seen so far every this many epochs")
    # These three commented-out flags are now replaced by --variant
    # parser.add_argument("--dist-fn", type=str, choices=["l2", "dotprod"], required=True)
    # parser.add_argument("--stage-classification", action="store_true", default=False)
    # parser.add_argument("--predict-lang-embs", action="store_true", default=False)
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=[
            "lang-dist-l2", "lang-dist-dotprod",
            "lang-reg",
            "stage-classif", "stage-reg"])
    parser.add_argument("--r3m", action="store_true", default=False)
    parser.add_argument("--img-net", action="store_true", default=False)
    parser.add_argument(
        "--unfrozen-mods", default="all",
        choices=[
            "all",  # whole net training
            "cnnlastlayer",  # only last layer of resnet or r3m training
            "adapters",  # add adapters to r3m
            "none"])  # debug: freeze entire network.
    parser.add_argument(
        "--adapter-compress-ratio", default=None, type=int)
    parser.add_argument("--two-stage-pp", type=int, choices=[0, 1, 2, 4, 6, 8, -1, -2], default=0)
    parser.add_argument("--override-lang", action="store_true", default=False)
    parser.add_argument(
        "--use-pred-stages", action="store_true", default=False,
        help=("Use pred_stage_nums (collected with automatic labeller) "
              "instead of lang_stage_nums (collected during scripted policy)"))

    # Flags only needed for rephrasing lang.
    parser.add_argument("--rephrasing-csv", type=str, default="")
    parser.add_argument(
        "--mean-rephrase-emb", action="store_true", default=False,
        help="Use the mean of all lang emb rephrases as the target.")
    parser.add_argument("--env", type=str, nargs="+", default=[])
    parser.add_argument("--realrobot-target-obj", type=str, default="")

    parser.add_argument(
        "--shuffle-demos", action="store_true", default=False,
        help=(
            "Whether to shuffle the demos in the hdf5 before taking the first"
            " dom1/2-num-demos-per-task"))
    args = parser.parse_args()

    if args.loss_type == "img-lang-time-contr":
        assert 0.0 <= args.tc_alpha <= 1.0
        assert args.tc_sim_fn is not None

    if args.rephrasing_csv != "":
        assert args.variant == "lang-reg"
        assert len(args.env) == 2

    if args.variant in ["lang-dist-l2", "lang-dist-dotprod", "stage-reg"]:
        if not isinstance(args.loss_arg_mult, float):
            print("setting loss arg mult to 1.0.")
            args.loss_arg_mult = 1.0
        assert args.loss_arg_mult > 0.0
    else:
        assert args.loss_arg_mult is None

    if args.unfrozen_mods != "all":
        assert args.r3m

    if args.unfrozen_mods == "adapters":
        assert args.r3m
        assert args.adapter_compress_ratio is not None

    args.out_dir = os.path.join(args.out_dir, get_timestamp_str())

    # Postprocess some args
    args.hdf5_kwargs_dom1 = {}
    if os.path.splitext(args.dom1_img_dir)[-1] == ".hdf5":
        args.dom1_task_idxs = exp_utils.create_task_indices_from_task_int_str_list(
            args.dom1_task_idxs, np.inf)
        if len(args.dom1_task_idxs) > 0:
            args.hdf5_kwargs_dom1['task_indices'] = args.dom1_task_idxs
        if args.dom1_num_demos_per_task is not None:
            args.hdf5_kwargs_dom1['max_demos_per_task'] = args.dom1_num_demos_per_task
    args.hdf5_kwargs_dom2 = {}
    if os.path.splitext(args.dom2_img_dir)[-1] == ".hdf5":
        args.dom2_task_idxs = exp_utils.create_task_indices_from_task_int_str_list(
            args.dom2_task_idxs, np.inf)
        if len(args.dom2_task_idxs) > 0:
            args.hdf5_kwargs_dom2['task_indices'] = args.dom2_task_idxs
        if args.dom2_num_demos_per_task is not None:
            args.hdf5_kwargs_dom2['max_demos_per_task'] = args.dom2_num_demos_per_task
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    train(args)
    plot_vals(os.path.join(args.out_dir, "data.csv"), args.out_dir, ["Val f1>0.8", "Val f1>0.5"])
    plot_vals(os.path.join(args.out_dir, "data.csv"), args.out_dir, ["Val loss"])
