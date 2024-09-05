from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    pairwise_diffs_matrix, calc_over_thresh_f1)
from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.torch.policies import MakeDeterministic
from rlkit.torch.alg.torch_rl_algorithm import TorchTrainer
from rlkit.util.misc_functions import tile_embs_by_batch_size
import rlkit.util.pytorch_util as ptu


class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,

            policy_lr=1e-3,
            policy_weight_decay=0,
            optimizer_class=optim.Adam,

            bc_batch_size=64,
            video_batch_size=None,

            bc_weight=1.0,
            task_encoder_weight=0.0,
            aux_task_weight=0.0,

            multitask=False,

            meta_batch_size=4,
            train_tasks=[],

            task_encoder=None,  # For example, BCZVideoEncoder
            task_embedding_type=None,

            finetune_lang_enc=False,
            lang_encoder=None,  # For example, DistilBERT
            target_emb_net=None,  # For example, Mlp

            # obs_key_to_dim_map={},
            emb_obs_keys=[],
            # observation_keys=[],

            task_emb_input_mode="concat_to_img_embs",
            policy_num_film_inputs=0,
            policy_film_input_order="",
            policy_loss_type="logp",
            gripper_loss_weight=None,
            gripper_loss_type=None,
            mmd_coefficient=0.0,

            # Phase 1-2 args
            policy_cnn_dist_fn="",
            dist_loss_arg_mult=None,
            xdomain_lang_embs_diff_mat=None,
            target_img_lang_coeff=None,
            img_lang_coeff_schedule="",
            transition1_epoch=10,
            policy_cnn_ckpt_unfrozen_mods=[],
            phase1_2_debug_mode=0,
            buf_idx_to_env_idx_map=None,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy

        # torch.autograd.set_detect_anomaly(True)
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.policy_params = list(self.policy.parameters())
        if task_encoder is not None:
            self.policy_params.extend(list(task_encoder.encoder.parameters()))
        if lang_encoder is not None:
            self.policy_params.extend(list(lang_encoder.parameters()))
        if target_emb_net is not None:
            self.policy_params.extend(list(target_emb_net.parameters()))

        self.policy_optimizer = self.optimizer_class(
            self.policy_params, **self.optimizer_kwargs)

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_batch_size = bc_batch_size
        self.bc_weight = bc_weight
        self.eval_policy = MakeDeterministic(self.policy)

        self.multitask = multitask

        self.meta_batch_size = meta_batch_size
        self.train_tasks = train_tasks

        self.task_encoder = task_encoder
        self.task_embedding_type = task_embedding_type
        self.task_encoder_weight = task_encoder_weight
        self.video_batch_size = video_batch_size  # num cached embs to retrieve
        self.aux_task_weight = aux_task_weight

        self.finetune_lang_enc = finetune_lang_enc
        self.lang_encoder = lang_encoder
        self.target_emb_net = target_emb_net

        # self.obs_key_to_dim_map = obs_key_to_dim_map
        self.emb_obs_keys = emb_obs_keys
        self.obs_key_to_obs_idx_pairs = {}
        # self.observation_keys = observation_keys

        self.task_emb_input_mode = task_emb_input_mode
        self.use_film = bool(task_emb_input_mode in [
            "film", "film_video_concat_lang", "film_lang_concat_video"])
        self.policy_num_film_inputs = policy_num_film_inputs
        self.policy_film_input_order = policy_film_input_order

        assert policy_loss_type in ["logp", "mse"]
        self.policy_loss_type = policy_loss_type
        if self.policy.gripper_policy_arch == "sep_head":
            self.gripper_loss_weight = gripper_loss_weight
            self.gripper_loss_type = gripper_loss_type

            if self.gripper_loss_type == "ce":
                self.gripper_loss_fn = nn.CrossEntropyLoss()
            elif self.gripper_loss_type == "mse":
                self.gripper_loss_fn = nn.MSELoss()
            else:
                raise NotImplementedError

        self.dom1_bsz = self.meta_batch_size * self.bc_batch_size
        # Phase 1-2 args
        self.img_lang_dist_learning = bool(
            target_img_lang_coeff is not None)
        self.policy_cnn_dist_fn = policy_cnn_dist_fn
        self.dist_loss_arg_mult = dist_loss_arg_mult
        self.xdomain_lang_embs_diff_mat = xdomain_lang_embs_diff_mat
        self.target_img_lang_coeff = target_img_lang_coeff
        self.img_lang_coeff_schedule = img_lang_coeff_schedule
        self.transition1_epoch = transition1_epoch
        self.policy_cnn_ckpt_unfrozen_mods = policy_cnn_ckpt_unfrozen_mods
        self.phase1_2_debug_mode = phase1_2_debug_mode
        self.froze_cnn_already = False
        if self.img_lang_dist_learning:
            self.alpha = 1.0  # Initial value; updated at each iteration/epoch.
        else:
            self.alpha = 0.0
        self.buf_idx_to_env_idx_map = buf_idx_to_env_idx_map
        self.mmd_coefficient = mmd_coefficient
        # we only want to freeze cnn once if there are elems in
        # self.policy_cnn_ckpt_unfrozen_mods

    def maybe_transform_target_embs(self, traj_emb_targets):
        if self.target_emb_net is not None:
            # traj_emb_targets.shape: (meta_bs, 768)
            traj_emb_targets = self.target_emb_net(traj_emb_targets)
        return traj_emb_targets

    def debug_get_var_vals(self):
        """Debug function to probe certain weights of networks"""
        if hasattr(self.policy, "gaussian_policy"):
            var_vals = {
                "layer4film": (
                    self.policy.gaussian_policy.cnn.layer4[0]
                    .film_blocks.film_blocks[0]
                    .gamma_net.last_fc.weight[0, :5]),
                "layer3film": (
                    self.policy.gaussian_policy.cnn.layer3[0]
                    .film_blocks.film_blocks[0]
                    .gamma_net.last_fc.weight[0, :5]),
                "layer4conv": (
                    self.policy.gaussian_policy.cnn.layer4[1]
                    .conv1.weight[0, 0, 0]),
                "layer3conv": (
                    self.policy.gaussian_policy.cnn.layer3[1]
                    .conv1.weight[0, 0, 0]),
                "gauss_policy_fc": (
                    self.policy.gaussian_policy.fc_layers[0].weight[0, :5]),
                "resid_fc": self.policy.fc_layers[0].weight[0, :5],
            }
        elif hasattr(self.policy.cnn, "module"):
            var_vals = {
                "layer4conv": (
                    self.policy.cnn.module.convnet.layer4[1]
                    .conv1.weight[0, 0, 0]),
                "layer3conv": (
                    self.policy.cnn.module.convnet.layer3[1]
                    .conv1.weight[0, 0, 0]),
                "fc": self.policy.fc_layers[0].weight[0, :5],
            }
            if (self.policy.cnn.module.convnet.layer4[1].conv_adapter
                    is not None):
                var_vals.update(dict(
                    layer4conv_adapter=(
                        self.policy.cnn.module.convnet.layer4[1]
                        .conv_adapter.down_conv.weight[0, 0, 0]),
                    layer3conv_adapter=(
                        self.policy.cnn.module.convnet.layer3[1]
                        .conv_adapter.down_conv.weight[0, 0, 0]),
                ))
        else:
            var_vals = {
                "layer4film": (
                    self.policy.cnn.layer4[0].film_blocks.film_blocks[0]
                    .gamma_net.last_fc.weight[0, :5]),
                "layer3film": (
                    self.policy.cnn.layer3[0].film_blocks.film_blocks[0]
                    .gamma_net.last_fc.weight[0, :5]),
                "layer4conv": self.policy.cnn.layer4[1].conv1.weight[0, 0, 0],
                "layer3conv": self.policy.cnn.layer3[1].conv1.weight[0, 0, 0],
                "fc": self.policy.fc_layers[0].weight[0, :5],
            }
        for name, val in var_vals.items():
            var_vals[name] = val.detach().cpu().numpy()
        return var_vals

    def run_bc_batch(self, batch, validation=False):
        prefix = "validation" if validation else "train"
        t, b, _ = batch['observations'].size()
        o = batch["observations"].view(t * b, -1)
        u = batch["actions"].view(t * b, -1)
        if self.policy.gripper_policy_arch == "sep_head":
            u_gripper = batch["gripper_actions"].view(t * b, -1)
            u_gripper = u_gripper.squeeze()  # Make 1-D for targets
        else:
            u_gripper = None

        losses = dict()

        if self.task_encoder is not None:
            if self.task_encoder.use_cached_embs:
                traj_emb_preds_dict = (
                    self.task_encoder.get_cached_embs_from_task_ids(
                        batch['task_indices'], self.bc_batch_size))
                traj_emb_targets = batch['target_traj_embs']
                traj_emb_targets = self.maybe_transform_target_embs(
                    traj_emb_targets)
                traj_emb_targets_tiled = tile_embs_by_batch_size(
                    traj_emb_targets, self.bc_batch_size)
                loss_kwargs = {}
            else:
                task_encoder_batch_dict = batch['task_encoder_batch_dict']
                # Process task_encoder_batch_dict
                for key, val in task_encoder_batch_dict.items():
                    # task_encoder_batch_dict[key].shape:
                    # (meta_bs, bs, 3, im_size * num_rows, im_size * num_cols)
                    task_encoder_batch_dict[key] = torch.cat(
                        [val[i] for i in range(val.shape[0])], dim=0)
                    # task_encoder_batch_dict[key].shape:
                    # (meta_bs * bs, 3, im_size * num_rows, im_size * num_cols)

                # Process traj_emb_targets (language)
                if (isinstance(self.task_encoder.encoder, ClipWrapper)
                        and not self.task_encoder.use_cached_embs):
                    task_lang_tokens = (
                        self.task_encoder.encoder.get_task_lang_tokens_matrix(
                            batch['task_indices']))
                    task_encoder_batch_dict["clip_lang"] = task_lang_tokens
                else:
                    traj_emb_targets = batch['target_traj_embs']

                    if self.finetune_lang_enc:
                        # traj_emb_targets contains the tokens, not the embs.
                        # We want to compute the embs
                        assert not self.lang_encoder.freeze
                        traj_emb_targets = self.lang_encoder(
                            traj_emb_targets.long())

                    traj_emb_targets = self.maybe_transform_target_embs(
                        traj_emb_targets)
                    if self.task_embedding_type in ["demo_lang", "mcil"]:
                        # traj_emb_targets: (mbs, latent_out_dim)
                        # --> (mbs * bs, latent_out_dim)
                        traj_emb_targets_tiled = tile_embs_by_batch_size(
                            traj_emb_targets, self.bc_batch_size)
                        # Check the order of the emb_obs_keys in
                        # init_emb_keys_and_obs_space(...)
                        if self.task_encoder.num_film_inputs > 0:
                            task_encoder_batch_dict["lang"] = (
                                traj_emb_targets_tiled)

                traj_emb_preds_dict = self.task_encoder(
                    task_encoder_batch_dict)

                loss_kwargs = {}
                if isinstance(self.task_encoder.encoder, ClipWrapper):
                    loss_kwargs['logit_scale_list'] = [
                        traj_emb_preds_dict.pop("logit_scale")]
                    traj_emb_targets = traj_emb_preds_dict.pop("lang")
                    traj_emb_targets = self.maybe_transform_target_embs(
                        traj_emb_targets)
                    traj_emb_targets_tiled = tile_embs_by_batch_size(
                        traj_emb_targets, self.bc_batch_size)

            # Order the embs as according to self.emb_obs_keys
            if len(self.emb_obs_keys) > 1:
                traj_emb_preds_list = []
                for emb_obs_key in self.emb_obs_keys:
                    if "video" in emb_obs_key:
                        vid_enc_mod_key = emb_obs_key.replace("_embedding", "")
                        traj_emb_preds_list.append(
                            traj_emb_preds_dict[vid_enc_mod_key])
            elif len(self.emb_obs_keys) == 1:
                if (self.task_embedding_type == "demo"
                        and "lang" in traj_emb_preds_dict):
                    traj_emb_preds_dict.pop("lang")
                assert len(traj_emb_preds_dict) == 1
                traj_emb_preds_list = list(traj_emb_preds_dict.values())
            else:
                raise NotImplementedError

            if self.task_embedding_type != "mcil":
                # t = time.time()
                losses['task_encoder_loss'] = (
                    self.task_encoder.loss_criterion.calc(
                        traj_emb_preds_list, traj_emb_targets, **loss_kwargs))
                # print("task enc loss computation", time.time() - t)

            if self.task_embedding_type == 'demo':
                emb_or_emb_list = traj_emb_preds_list
            elif self.task_embedding_type == 'demo_lang':
                if (self.task_encoder.num_film_inputs == 0 or
                        self.task_emb_input_mode in [
                            "film_video_concat_lang",
                            "film_lang_concat_video"]):
                    if self.policy_film_input_order == "lv":
                        emb_or_emb_list = (
                            [traj_emb_targets_tiled] + traj_emb_preds_list)
                    elif self.policy_film_input_order in ["vl", ""]:
                        emb_or_emb_list = (
                            traj_emb_preds_list + [traj_emb_targets_tiled])
                elif self.task_encoder.num_film_inputs > 0:
                    # Language (targets) was already added to video encoder
                    # via film.
                    emb_or_emb_list = traj_emb_preds_list
            elif self.task_embedding_type == 'mcil':
                emb_or_emb_list = None
        elif self.task_embedding_type in ["lang", "onehot"]:
            emb_or_emb_list = None
        else:
            print("self.task_embedding_type", self.task_embedding_type)
            raise NotImplementedError

        if self.task_embedding_type == "mcil":
            assert len(traj_emb_preds_list) == 1
            self.modality_to_emb_map = {
                "lang": [traj_emb_targets_tiled],
                "video": traj_emb_preds_list,
            }

            # These loss terms average over all modalities
            losses['logp_loss'] = 0.0
            losses['mse_loss'] = 0.0

            if self.policy.gripper_policy_arch == "sep_head":
                losses['gripper_loss'] = 0.0

            for modality, film_inputs in self.modality_to_emb_map.items():
                policy_losses, stats = self.compute_losses(
                    o, film_inputs, u, u_gripper,
                    prefix=prefix)

                # add a prefix to keys in policy_losses when adding
                # to losses dict.
                for key, val in policy_losses.items():
                    new_key = f"{modality}_{key}"
                    losses[new_key] = val

                losses['logp_loss'] += policy_losses['logp_loss'] * (
                    1 / len(self.modality_to_emb_map))
                losses['mse_loss'] += policy_losses['mse_loss'] * (
                    1 / len(self.modality_to_emb_map))
                losses['mmd_loss'] = policy_losses['mmd_loss'] * ( #is this necessary
                    1 / len(self.modality_to_emb_map))

                if self.policy.gripper_policy_arch == "sep_head":
                    losses['gripper_loss'] += policy_losses['gripper_loss'] * (
                        1 / len(self.modality_to_emb_map))
        else:
            o, emb = self.process_obs_and_emb(o, emb_or_emb_list)
            # print("train:", torch.norm(o[:,-384:], dim=1)[:10])
            film_inputs = emb if self.use_film else None

            extra_loss_kwargs = {}
            if (self.img_lang_dist_learning
                    and self.phase1_2_debug_mode not in [1, 3]
                    and self.alpha > 0.0):
                # make lang_idx1 shape (meta_bsz, bsz) --> (meta_bsz * bsz)
                extra_loss_kwargs['lang_idx1'] = torch.cat(
                    list(batch['lang_idx1']), dim=0).long()
                extra_loss_kwargs['lang_idx2'] = torch.cat(
                    list(batch['lang_idx2']), dim=0).long()
            if self.mmd_coefficient > 0:
                extra_loss_kwargs['batch_idxs'] = batch['task_indices']
            policy_losses, stats = self.compute_losses(
                o, film_inputs, u, u_gripper,
                prefix=prefix,
                **extra_loss_kwargs)
            losses.update(policy_losses)

        if self.policy_loss_type == "logp":
            losses['policy_loss'] = losses['logp_loss']
        elif self.policy_loss_type == "mse":
            losses['policy_loss'] = losses['mse_loss']

        if self.policy.gripper_policy_arch == "sep_head":
            losses['policy_loss'] += (
                self.gripper_loss_weight * losses['gripper_loss'])

        return losses, stats

    def compute_losses(
            self, o, film_inputs, u, u_gripper=None, lang_idx1=None,
            lang_idx2=None, prefix='train', batch_idxs=None):
        policy_losses = {}

        policy_kwargs = {}
        if self.use_film:
            policy_kwargs['film_inputs'] = film_inputs
        if ((self.img_lang_dist_learning
                and self.phase1_2_debug_mode not in [1, 3]
                and self.alpha > 0.0) or self.mmd_coefficient > 0):
            policy_kwargs['output_cnn_embs_in_aux'] = True
        dist, policy_stats_dict, aux_outputs = self.policy(o, **policy_kwargs)

        pred_u, _ = dist.rsample_and_logprob()
        stats = dist.get_diagnostics()
        stats.update(policy_stats_dict)

        # get gripper output from aux_outputs
        if self.policy.gripper_policy_arch == "sep_head":
            gripper_class_preds = torch.round(
                1 + aux_outputs["preds"]["gripper_actions"]).squeeze()
            if self.gripper_loss_type == "ce":
                gripper_preds_for_loss = aux_outputs["preds"]["gripper_logits"]
                u_gripper = u_gripper.long()
            elif self.gripper_loss_type == "mse":
                gripper_preds_for_loss = (
                    1.0 + aux_outputs["preds"]["gripper_actions"]).squeeze()
            else:
                raise NotImplementedError
            policy_losses["gripper_loss"] = self.gripper_loss_fn(
                gripper_preds_for_loss, u_gripper)
            stats["gripper accuracy"] = float(ptu.get_numpy(
                torch.mean((u_gripper == gripper_class_preds).float())))
            # Proportion of predicted actions that are close or open gripper.
            stats["gripper close prop"] = float(ptu.get_numpy(
                torch.mean((gripper_class_preds == 0).float())))
            stats["gripper open prop"] = float(ptu.get_numpy(
                torch.mean((gripper_class_preds == 2).float())))

        if len(self.policy.aux_tasks) > 0 and isinstance(aux_outputs, dict):
            policy_losses.update(aux_outputs['losses'])

        logp_loss = -dist.log_prob(u,)
        if self.mmd_coefficient > 0:
            batch_idxs = batch_idxs.cpu()
            mask = np.zeros_like(batch_idxs)
            for i in range(len(batch_idxs)):
                if self.buf_idx_to_env_idx_map[int(batch_idxs[i])] >= 1:
                    # the sim env always has env_idx 0
                    # real world target is usually 1.
                    # If exists, real world prior is usually 2
                    mask[i] = 1

            if np.sum(mask) == 0 or np.sum(1 - mask) == 0:
                policy_losses['mmd_loss'] = torch.tensor(
                    0.0, requires_grad=True)
            else:
                dom1_avg = torch.zeros_like(aux_outputs[
                    'cnn_emb_for_dist_learning'][0:self.bc_batch_size])
                dom2_avg = torch.zeros_like(aux_outputs[
                    'cnn_emb_for_dist_learning'][0:self.bc_batch_size])
                for i in range(len(mask)):
                    start_idx = i * self.bc_batch_size
                    end_idx = (i + 1) * self.bc_batch_size
                    if mask[i] == 1:
                        dom1_avg += aux_outputs['cnn_emb_for_dist_learning'][
                            start_idx:end_idx]
                    else:
                        dom2_avg += aux_outputs['cnn_emb_for_dist_learning'][
                            start_idx:end_idx]
                dom1_avg /= np.sum(mask)
                dom2_avg /= np.sum(1 - mask)
                dom1_avg = torch.mean(dom1_avg, dim=0)
                dom2_avg = torch.mean(dom2_avg, dim=0)
                policy_losses['mmd_loss'] = torch.norm(dom1_avg - dom2_avg)

        # IF doing img-lang distance learning, then don't use domain2 images
        # to predict loss.
        if self.img_lang_dist_learning and self.alpha > 0.0:
            def mask_x(mask, x):
                singleton_dims = tuple([1] * (len(x.shape) - 1))
                mask = mask.reshape(mask.shape[0], *singleton_dims)
                return mask * x
            assert pred_u.shape[0] == 2 * self.dom1_bsz
            mask = ptu.from_numpy(np.concatenate([
                np.ones(self.dom1_bsz), np.zeros(self.dom1_bsz)], axis=0))
            pred_u = mask_x(mask, pred_u)
            u = mask_x(mask, u)  # was already masked but doing it again
            logp_loss = mask_x(mask, logp_loss)
        else:
            # It's possible that img_lang_dist_learning loses its dom2 data
            # in batch when phase 1 in merged phase 1-2 is complete.
            assert pred_u.shape[0] == self.dom1_bsz

        mse_loss = nn.MSELoss()(pred_u, u)

        policy_losses.update(dict(
            logp_loss=logp_loss.mean(),
            mse_loss=mse_loss,
        ))

        if self.img_lang_dist_learning and self.alpha > 0.0:
            # Correct for the loss terms taking mean over ignored entries
            policy_losses['logp_loss'] *= 2
            policy_losses['mse_loss'] *= 2

            if self.phase1_2_debug_mode not in [1, 3]:
                out = aux_outputs['cnn_emb_for_dist_learning']
                assert out.shape[0] == 2 * self.dom1_bsz
                if self.froze_cnn_already:
                    # DEBUG: detach if done with img_lang_dist_learning phase.
                    out = out.detach()
                out1 = out[:self.dom1_bsz]
                out2 = out[self.dom1_bsz:]
                preds = pairwise_diffs_matrix(
                    out1, out2, self.policy_cnn_dist_fn)
                idxs = torch.cartesian_prod(lang_idx1, lang_idx2)
                # gives us a list of tuples of the coordinates from which
                # we want to pluck the xdomain_lang_embs_diff_mat
                targets = self.xdomain_lang_embs_diff_mat[
                    idxs[:, 0], idxs[:, 1]].reshape(
                        self.dom1_bsz, self.dom1_bsz)
                policy_losses[f"{prefix}_img_lang_dist_loss"] = F.mse_loss(
                    self.dist_loss_arg_mult * preds,
                    self.dist_loss_arg_mult * targets)
                stats[f"{prefix} img lang f1>0.5"] = calc_over_thresh_f1(
                    preds.detach(), targets.detach(), thresh=0.5)
                stats[f"{prefix} img lang f1>0.8"] = calc_over_thresh_f1(
                    preds.detach(), targets.detach(), thresh=0.8)

        return policy_losses, stats

    def split_emb_from_obs(self, obs):
        assert len(self.emb_obs_keys) == 1
        assert self.task_embedding_type in ["lang", "onehot"]

        emb_start_idx, emb_end_idx = self.policy.obs_key_to_obs_idx_pairs[
            self.emb_obs_keys[0]]
        emb = obs[:, emb_start_idx:emb_end_idx]
        if emb_end_idx == obs.shape[-1]:
            obs_without_emb = obs[:, :emb_start_idx]
        else:
            obs_without_emb = torch.cat(
                [obs[:, :emb_start_idx], obs[:, emb_end_idx:]], dim=-1)
        return obs_without_emb, emb

    def process_obs_and_emb(self, o, emb_or_emb_list):
        if self.task_embedding_type in ["lang", "onehot"]:
            if self.use_film:
                o, emb = self.split_emb_from_obs(o)
                if self.finetune_lang_enc:
                    # emb contains the tokens, not the actual embs.
                    # We want to compute the embs
                    assert not self.lang_encoder.freeze
                    emb = self.lang_encoder(emb.long())
            else:
                emb = None
        elif self.task_embedding_type == 'demo':
            # assert torch.is_tensor(emb_or_emb_list)
            emb = emb_or_emb_list
        elif self.task_embedding_type == 'demo_lang':
            assert isinstance(emb_or_emb_list, list)
            if self.task_emb_input_mode in [
                    "film_video_concat_lang", "film_lang_concat_video"]:
                assert len(emb_or_emb_list) == 2
                assert self.policy_film_input_order in ["vl", ""]
                video_emb = emb_or_emb_list[0]  # Video
                lang_emb = emb_or_emb_list[1]
                # In accordance with the order specified in the creation of
                # emb_or_emb_list right before this function is called

                if self.task_emb_input_mode == "film_video_concat_lang":
                    emb = video_emb
                elif self.task_emb_input_mode == "film_lang_concat_video":
                    emb = lang_emb
            elif self.policy_num_film_inputs in [2, 3]:
                emb = emb_or_emb_list
            elif self.policy_num_film_inputs in [0, 1]:
                # Might be a slowdown to do two cats instead of one.
                emb = torch.cat(emb_or_emb_list, dim=1)
            else:
                raise NotImplementedError

        if self.use_film:
            if self.policy_num_film_inputs == 1 and torch.is_tensor(emb):
                # Make into a singleton list
                emb = [emb]
            assert isinstance(emb, list)

            if self.task_emb_input_mode == "film_video_concat_lang":
                o = torch.cat([o, lang_emb], dim=1)
            elif self.task_emb_input_mode == "film_lang_concat_video":
                o = torch.cat([o, video_emb], dim=1)
        elif self.task_embedding_type in ["demo", "demo_lang"]:
            # not self.use_film
            if isinstance(emb, list):
                # Singleton lists end up as emb[0] after cat.
                emb = torch.cat(emb, dim=1)
            o = torch.cat([o, emb], dim=1)

        return o, emb

    def val_stats(self, batch):
        with torch.no_grad():
            losses, test_stats = self.run_bc_batch(batch, validation=True)
            stats = {}
            if (self.img_lang_dist_learning
                    and self.phase1_2_debug_mode not in [1, 3]
                    and self.alpha > 0.0):
                stats['Validation Img-Lang Dist Loss'] = ptu.get_numpy(
                    losses['validation_img_lang_dist_loss'])
            self.eval_statistics.update(stats)
            self.eval_statistics.update(test_stats)

    def train_from_torch(self, batch, epoch_num=None):
        # import ipdb; ipdb.set_trace()
        losses, train_stats = self.run_bc_batch(batch)
        self.eval_statistics.update(train_stats)

        train_policy_loss = losses['policy_loss'] * self.bc_weight

        if (self.task_encoder is not None
                and self.task_embedding_type != "mcil"):
            train_policy_loss += (
                losses['task_encoder_loss'] * self.task_encoder_weight)

        if len(self.policy.aux_tasks) > 0 and self.aux_task_weight != 0.0:
            aux_loss_sum = 0.0
            for aux_task in self.policy.aux_tasks:
                aux_loss_sum += losses[aux_task]
            train_policy_loss += aux_loss_sum * self.aux_task_weight

        if self.img_lang_dist_learning:
            self.alpha = self.get_policy_loss_coeff(epoch_num)
            if self.alpha > 0.0:  # DEBUG added alpha > 0.0 cond
                train_policy_loss = (
                    (1 - self.alpha) * train_policy_loss
                    + self.alpha * losses['train_img_lang_dist_loss'])

        if self.mmd_coefficient > 0:
            losses['policy_loss'] += losses['mmd_loss'] * self.mmd_coefficient

        # var_vals_pre_step = self.debug_get_var_vals()
        self.policy_optimizer.zero_grad()
        train_policy_loss.backward()
        self.policy_optimizer.step()

        if self.img_lang_dist_learning:
            if (epoch_num == self.transition1_epoch
                    and not self.froze_cnn_already):
                if len(self.policy_cnn_ckpt_unfrozen_mods) > 0:
                    self.policy.freeze_cnn()
                    if "cnnlastlayer" in self.policy_cnn_ckpt_unfrozen_mods:
                        self.policy.unfreeze_last_layer()
                    if "film" in self.policy_cnn_ckpt_unfrozen_mods:
                        self.policy.unfreeze_film_blocks()
                    if "cnnlayers" in self.policy_cnn_ckpt_unfrozen_mods:
                        self.policy.unfreeze_all_layers()
                    # assert self.alpha == 0.0

                    # REDEFINE optimizer, otherwise nothing actually gets
                    # frozen even though they are marked as
                    # requires_grad = False
                    self.policy_optimizer = self.optimizer_class(
                        filter(lambda p: p.requires_grad, self.policy_params),
                        **self.optimizer_kwargs)
                else:
                    # If no policy_cnn_ckpt_unfrozen_mods specified, then
                    # just unfreeze film.
                    self.policy.unfreeze_film_blocks()
                self.froze_cnn_already = True

        # for name, param in self.policy.named_parameters():
        #     if param.requires_grad:
        #         print(name, "requires grad")

        # var_vals_after_step = self.debug_get_var_vals()

        # print("checking to see if weights changed after step")
        # for name in var_vals_pre_step:
        #     diff = var_vals_pre_step[name] - var_vals_after_step[name]
        #     if np.linalg.norm(diff) > 1e-9:
        #         print(f"{name} updated, with diff", diff)
        # print("done checking weights")

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            stats = {
                "Train Logprob Loss": ptu.get_numpy(losses['logp_loss']),
                "Train MSE": ptu.get_numpy(losses['mse_loss']),
                "train_policy_loss": ptu.get_numpy(train_policy_loss),
            }

            if self.policy.gripper_policy_arch == "sep_head":
                stats['Train Gripper Loss'] = ptu.get_numpy(
                    losses['gripper_loss'])

            if self.img_lang_dist_learning:
                if self.phase1_2_debug_mode not in [1, 3] and self.alpha > 0.0:
                    stats['Train Img-Lang Dist Loss'] = ptu.get_numpy(
                        losses['train_img_lang_dist_loss'])
                stats['img-lang alpha'] = self.alpha

            if self.task_embedding_type == "mcil":
                loss_category_to_str_map = {
                    "logp_loss": "Logprob Loss",
                    "mse_loss": "MSE",
                }
                for modality in self.modality_to_emb_map:
                    for loss_category in ["logp_loss", "mse_loss"]:
                        key = (
                            f"Train {modality} "
                            f"{loss_category_to_str_map[loss_category]}")
                        stats[key] = ptu.get_numpy(
                            losses[f"{modality}_{loss_category}"])

            if (self.task_encoder is not None
                    and self.task_embedding_type != "mcil"):
                stats['Train Task encoder loss'] = ptu.get_numpy(
                    losses['task_encoder_loss'])

            if self.mmd_coefficient > 0:
                stats['Train MMD Loss'] = ptu.get_numpy(losses['mmd_loss'])

            if (len(self.policy.aux_tasks) > 0
                    and self.policy.aux_to_feed_fc in ["none", "preds"]):
                for aux_task in self.policy.aux_tasks:
                    stats[f"Train {aux_task} loss"] = ptu.get_numpy(
                        losses[aux_task])

            self.eval_statistics.update(stats)

        self._n_train_steps_total += 1

    def get_policy_loss_coeff(self, epoch_num):
        """
        Only called when doing some sort of scheduling.
        For now, this is when self.img_lang_dist_learning is on.
        """
        if self.img_lang_coeff_schedule == "const":
            alpha = self.target_img_lang_coeff
        elif self.img_lang_coeff_schedule == "const-const":
            alpha = 1.0
            if epoch_num >= self.transition1_epoch:
                alpha = self.target_img_lang_coeff
        elif self.img_lang_coeff_schedule == "linrampdown-const":
            alpha = self.target_img_lang_coeff + (
                1 - self.target_img_lang_coeff) * max(1 - 0.01 * epoch_num, 0)
        elif self.img_lang_coeff_schedule == "const-linrampdown-const":
            alpha = 1.0
            if epoch_num >= self.transition1_epoch:
                alpha = self.target_img_lang_coeff + (
                    (1 - self.target_img_lang_coeff)
                    * max(1 - 0.01 * (epoch_num - self.transition1_epoch), 0))
        elif self.img_lang_coeff_schedule == "const-tanh":
            alpha = 1.0
            if epoch_num >= self.transition1_epoch:
                alpha = 0.5 * (
                    ((1 - self.target_img_lang_coeff)
                        * np.tanh((125 - epoch_num) / 25))
                    + (1 + self.target_img_lang_coeff)
                )
        else:
            raise NotImplementedError
        return alpha

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [self.policy, ]
        if self.task_encoder is not None:
            nets.append(self.task_encoder.encoder)
        if self.finetune_lang_enc:
            nets.append(self.lang_encoder)
        if self.target_emb_net is not None:
            nets.append(self.target_emb_net)
        return nets

    def get_snapshot(self):
        if hasattr(self.policy.cnn, "clip"):
            policy_state_dict = self.policy.state_dict()
            non_clip_policy_state_dict = dict([
                (k, v) for k, v in policy_state_dict.items()
                if "cnn.clip" not in k])
            snapshot = dict(
                policy=non_clip_policy_state_dict,
                # clip=self.policy.cnn.clip.module.state_dict(),
            )
        elif (hasattr(self.policy.cnn, "module")
                and hasattr(self.policy.cnn.module, "convnet")):
            # R3M
            policy_state_dict = self.policy.state_dict()
            non_r3m_policy_state_dict = dict([
                (k, v) for k, v in policy_state_dict.items()
                if "cnn.module" not in k])
            snapshot = dict(
                policy=non_r3m_policy_state_dict,
            )
        else:
            snapshot = dict(policy=self.policy)
        if self.task_encoder is not None:
            snapshot.update(task_encoder=self.task_encoder)
        if self.target_emb_net is not None:
            snapshot.update(target_emb_net=self.target_emb_net)
        if self.lang_encoder is not None:
            snapshot.update(lang_encoder=self.lang_encoder)
        return snapshot
