import torch
import torch.nn.functional as F

from rlkit.util.misc_functions import (
    expand_targets_to_pred_shape, l2_unit_normalize)
import rlkit.util.pytorch_util as ptu


class bcz_cosine_distance_loss_fn:
    def calc(self, pred, target, logit_scale=None):
        self.verify_pred_and_target(pred, target)
        target = expand_targets_to_pred_shape(pred, target)
        return 1 - (torch.mean(torch.sum(pred * target, dim=1)))

    def verify_pred_and_target(self, pred, target):
        def is_close_to(y_hat, y, eps):
            assert abs(y - y_hat) <= eps

        avg_pred_norm = torch.mean(torch.norm(pred, dim=1)).item()
        avg_target_norm = torch.mean(torch.norm(target, dim=1)).item()
        is_close_to(avg_pred_norm, 1.0, 1e-3)
        is_close_to(avg_target_norm, 1.0, 1e-3)


class clip_contrastive_loss_fn:
    def __init__(self, temp):
        assert temp > 0.0
        self.temp = temp

    def calc(self, z_preds, z_targets, logit_scale=None):
        z_targets = expand_targets_to_pred_shape(z_preds, z_targets)

        z_preds = l2_unit_normalize(z_preds)
        z_targets = l2_unit_normalize(z_targets)

        if logit_scale is not None:
            assert self.temp == 1.0
            sim_mat = logit_scale * (z_preds @ z_targets.t())
        else:
            sim_mat = (z_preds @ z_targets.t()) / self.temp
        labels = ptu.tensor(torch.arange(len(sim_mat)))

        loss1 = F.cross_entropy(sim_mat, labels)
        loss2 = F.cross_entropy(sim_mat.t(), labels)
        loss = 0.5 * (loss1 + loss2)
        return loss


class metabatch_cross_ent_fn:
    def __init__(self, temp, meta_batch_size):
        assert temp > 0.0
        self.temp = temp
        self.meta_batch_size = meta_batch_size

    def calc(self, z_embs, z_label_embs, logit_scale=None):
        """
        Assumes z_label_embs is already in ``unique task idx''
        format, not minibatch format
        """
        n = z_embs.shape[0]
        num_label_repeats = n // self.meta_batch_size

        z_embs = l2_unit_normalize(z_embs)
        z_label_embs = l2_unit_normalize(z_label_embs)

        if logit_scale is not None:
            assert self.temp == 1.0
            logits = logit_scale * (z_embs @ z_label_embs.t())
        else:
            # Use temperature argument
            logits = (z_embs @ z_label_embs.t()) / self.temp

        assert logits.shape == (
            n, self.meta_batch_size), f"logits.shape {logits.shape}"
        labels = ptu.tensor(
            torch.tensor([i // num_label_repeats for i in range(n)]))

        loss = F.cross_entropy(logits, labels)
        return loss


class multi_emb_loss_fn:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def calc(self, z_embs_list, z_label_embs, logit_scale_list=None):
        weighted_avg_loss = 0.0
        num_losses = len(z_embs_list)
        if logit_scale_list is None:
            logit_scale_list = [None] * num_losses
        assert len(logit_scale_list) == num_losses
        for z_embs, logit_scale in zip(z_embs_list, logit_scale_list):
            loss = self.loss_fn.calc(z_embs, z_label_embs, logit_scale)
            weighted_avg_loss += (1 / num_losses) * loss
        return weighted_avg_loss
