from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from r3m.models.models_r3m import R3M
from rlkit.torch.networks import (
        Mlp, CNN
)
from rlkit.torch.networks.beta_vae import BetaVAE
from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.torch.networks.film import FiLMAttnNet
from rlkit.torch.networks.image_augmentations import create_aug_transform_fns
from rlkit.torch.networks.resnet import SpatialSoftmax
from rlkit.torch.pretrained_models.language_models import LONGEST_SENTENCE_LEN
from rlkit.torch.policies.base import TorchStochasticPolicy
from rlkit.util.distributions import MultivariateDiagonalNormal
import rlkit.util.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class GaussianCNNPolicy(CNN, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, output_stage="last_activations")
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)


class GaussianStandaloneCNNPolicy(TorchStochasticPolicy):
    def __init__(
            self,
            cnn,
            image_size,  # (h, w, c)
            hidden_sizes,
            obs_dim,
            action_dim,
            freeze_policy_cnn,
            added_fc_input_size=0,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            hidden_activation=nn.ReLU(),
            aug_transforms=[],
            image_augmentation_padding=4,
            rnd_erase_prob=0.0,
            fc_normalization_type='none',
            output_activation=torch.tanh,
            batchnorm=False,
            cnn_output_stage="",
            use_spatial_softmax=False,
            film_emb_dim_list: List[int] = [],  # [] == don't use film.
            num_film_inputs: int = 0,  # number of separate embeddings.
            # There will be `num_film_inputs` film blocks for each BasicBlock.
            use_film_attn=False,
            film_attn_hidden_sizes=[],
            state_obs_dim=0,
            emb_obs_keys=[],
            aux_tasks=[],  # Any other aux tasks that need to be performed.
            aux_obs_bounds={},  # dict of dicts
            # [aux_task] --> dict(["hi"/"lo" --> np.array])
            observation_keys=[],
            obs_key_to_dim_map={},
            aux_to_feed_fc="none",  # Whether or not to pass aux outputs
            # as additional fc inputs.
            lang_emb_obs_is_tokens=False,  # True when we work with tokens
            # b/c lang_enc is finetuned.
            gripper_policy_arch="ac_dim",
            gripper_loss_type=None,
            # Phase 1-2
            output_dist_learning_cnn_embs_in_aux=False,
            **kwargs,  # Will most likely be unused
    ):
        super(GaussianStandaloneCNNPolicy, self).__init__()
        self.cnn = cnn
        self.image_size = image_size
        self.conv_input_length = np.prod(image_size)
        self.added_non_aux_fc_input_size = added_fc_input_size
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        self.hidden_activation = hidden_activation
        self.aug_transforms = aug_transforms
        self.im_aug_pad = image_augmentation_padding
        self.rnd_erase_prob = rnd_erase_prob
        self.fc_normalization_type = fc_normalization_type
        self.output_activation = output_activation
        self.batchnorm = batchnorm
        self.cnn_output_stage = cnn_output_stage
        self.use_spatial_softmax = use_spatial_softmax
        self.film_emb_dim_list = film_emb_dim_list
        if len(self.film_emb_dim_list) > 0:
            self.num_film_inputs = self.cnn.num_film_inputs
        elif len(self.film_emb_dim_list) == 0:
            self.num_film_inputs = 0
        self.state_obs_dim = state_obs_dim
        if use_film_attn:
            self.film_attn_net = FiLMAttnNet(
                state_obs_dim=self.state_obs_dim,
                num_film_inputs=num_film_inputs,
                emb_obs_keys=emb_obs_keys,
                hidden_sizes=film_attn_hidden_sizes,
                output_activation=nn.Softmax(dim=-1),
            )
        else:
            self.film_attn_net = None
        self.aux_tasks = aux_tasks
        self.aux_obs_bounds = aux_obs_bounds
        self.observation_keys = observation_keys
        self.obs_key_to_dim_map = obs_key_to_dim_map
        self.aux_to_feed_fc = aux_to_feed_fc
        self.lang_emb_obs_is_tokens = lang_emb_obs_is_tokens
        self.gripper_policy_arch = gripper_policy_arch
        self.gripper_loss_type = gripper_loss_type
        self.output_dist_learning_cnn_embs_in_aux = (
            output_dist_learning_cnn_embs_in_aux)
        if isinstance(self.cnn, ClipWrapper):
            self.cnn_out_dim = self.cnn.visual_outdim
        elif hasattr(self.cnn, "module") and isinstance(self.cnn.module, R3M):
            self.cnn_out_dim = self.cnn.module.outdim
        elif isinstance(self.cnn, torchvision.models.resnet.ResNet):
            self.cnn_out_dim = self.cnn.fc.out_features
            self.imgnet_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif isinstance(self.cnn, BetaVAE):
            self.cnn_out_dim = self.cnn.latent_dim
        else:
            test_mat = torch.zeros(
                1,
                image_size[2],
                image_size[0],
                image_size[1],
            )

            test_mat_cnn_kwargs = {}
            if len(self.film_emb_dim_list) > 0:
                test_film_embs = [
                    torch.zeros(1, film_emb_dim)
                    for film_emb_dim in self.film_emb_dim_list]
                if self.film_attn_net is not None:
                    test_film_embs, _ = self.film_attn_net(
                        torch.zeros(1, self.state_obs_dim), test_film_embs)
                test_mat_cnn_kwargs.update(film_inputs=test_film_embs)
            test_mat = self.cnn(
                test_mat, output_stage=self.cnn_output_stage,
                **test_mat_cnn_kwargs)
            self.cnn_out_dim = np.prod(test_mat.shape[1:])

            if self.use_spatial_softmax:
                print("test_mat.shape", test_mat.shape)
                self.spatial_softmax = SpatialSoftmax(
                    test_mat.shape[2], test_mat.shape[3], test_mat.shape[1])
                self.cnn_out_dim = 2 * test_mat.shape[1]
            print("self.cnn_out_dim (after potential spatial softmax)",
                  self.cnn_out_dim)

        if self.batchnorm:
            self.bn_layer = nn.BatchNorm1d(
                self.cnn_out_dim + self.added_non_aux_fc_input_size)

        self.create_obs_key_to_obs_idx_pair_map()

        if len(self.aux_tasks) > 0:
            # Needs to be done after self.cnn_out_dim is calculated.
            if self.aux_to_feed_fc != "ground_truths":
                # Not training any networks if simply passing
                # ground truth values into policy.
                self.aux_task_to_loss_map = dict([
                    (aux_task, nn.L1Loss(reduction='mean'))
                    for aux_task in self.aux_tasks])
                self.aux_task_to_net_map = self.init_aux_task_nets()

            self.aux_task_total_dim = np.sum(
                [self.obs_key_to_dim_map[aux_task]
                 for aux_task in self.aux_tasks])

        if len(self.aux_tasks) > 0 and self.aux_to_feed_fc in [
                "preds", "ground_truths"]:
            self.added_fc_input_size = (
                self.added_non_aux_fc_input_size + self.aux_task_total_dim)
        else:
            self.added_fc_input_size = self.added_non_aux_fc_input_size

        if self.gripper_policy_arch == "ac_dim":
            gaussian_action_dim = action_dim
        elif self.gripper_policy_arch == "sep_head":
            gaussian_action_dim = action_dim - 1
            self.gripper_ac_idx = 6
            # Based on it being the 2nd to last index in minibullet envs.

            if self.gripper_loss_type == "ce":
                gripper_fc_out_dim = 3  # output over 3 classes (-1, 0, 1)
            elif self.gripper_loss_type == "mse":
                gripper_fc_out_dim = 1
            else:
                raise NotImplementedError

            self.gripper_fc = Mlp(
                hidden_sizes=[32], output_size=gripper_fc_out_dim,
                input_size=self.cnn_out_dim + self.added_non_aux_fc_input_size,
                init_w=1e-3,)  # cross entropy loss for gripper
        else:
            raise NotImplementedError

        self.fc_layers, self.fc_norm_layers, self.last_fc = (
            ptu.initialize_fc_layers(
                hidden_sizes, gaussian_action_dim,
                self.fc_normalization_type, self.cnn_out_dim,
                self.added_fc_input_size, init_w))

        transf_kwargs = {
            "image_size": self.image_size,
            "im_aug_pad": self.im_aug_pad,
            "rnd_erase_prob": self.rnd_erase_prob,
            "aug_transforms": self.aug_transforms,
        }
        self.aug_transform_fns = create_aug_transform_fns(transf_kwargs)

        if freeze_policy_cnn:
            self.freeze_cnn()

        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(
                    last_hidden_size, gaussian_action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(gaussian_action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, input_obs, film_inputs=[], output_cnn_embs_in_aux=False):
        # TODO(cleanup): see if we can combine `output_cnn_embs_in_aux`
        # with `self.output_dist_learning_cnn_embs_in_aux`
        conv_input = input_obs.narrow(
            start=0,
            length=self.conv_input_length,
            dim=1).contiguous()

        # from PIL import Image
        # imarr = conv_input[::50]
        # imarr = np.uint8(255 * imarr.reshape(imarr.shape[0], 3, 128, 128).permute(0, 2, 3, 1).cpu())
        # imarr = np.concatenate(list(imarr), axis=0)
        # im = Image.fromarray(imarr)
        # im.save("202310102359.png")
        # quit()

        if len(self.aux_tasks) == 0:
            extra_non_aux_fc_input = input_obs.narrow(
                start=self.conv_input_length,
                length=self.added_non_aux_fc_input_size,
                dim=1,
            )
            assert (
                conv_input.shape[1] + self.added_non_aux_fc_input_size
                == input_obs.shape[1])
        elif len(self.aux_tasks) > 0:
            assert (
                conv_input.shape[1] + self.added_non_aux_fc_input_size +
                self.aux_task_total_dim == input_obs.shape[1])

        # (B, 6912) --> (B, 3, 48, 48)
        B, _ = conv_input.shape
        H, W, C = self.image_size
        conv_input = torch.reshape(conv_input, (B, C, H, W))

        if conv_input.shape[0] > 1:
            for aug_transform_fn in self.aug_transform_fns:
                # h.shape[0] > 1 ensures we apply this only during training
                conv_input = aug_transform_fn(conv_input)
        stats_dict = {}
        if self.film_attn_net is not None:
            film_inputs, film_attn_stats_dict = self.film_attn_net(
                extra_non_aux_fc_input, film_inputs)
            stats_dict.update(film_attn_stats_dict)

        if self.output_dist_learning_cnn_embs_in_aux:
            assert self.cnn_output_stage == "conv_channels"
            cnn_embs_dict = self.cnn(
                conv_input, output_stage="last-and-conv_channels",
                film_inputs=film_inputs)
            cnn_embs = cnn_embs_dict['conv_channels']
            cnn_emb_for_dist_learning = cnn_embs_dict['last']
        elif isinstance(self.cnn, ClipWrapper):
            assert self.cnn.freeze
            # train_mode == False so we don't do image aug or grad updates
            # on clip
            cnn_embs = self.cnn(conv_input, texts=None, train_mode=False)
        elif isinstance(self.cnn, torchvision.models.resnet.ResNet):
            conv_input = self.imgnet_transform(conv_input)
            cnn_embs = self.cnn(conv_input)
        elif hasattr(self.cnn, "module") and isinstance(self.cnn.module, R3M):
            try:
                cnn_embs = self.cnn(
                    conv_input,
                    obs_shape=torch.tensor(conv_input.shape[-3:]).tolist())
            except:
                cnn_embs = self.cnn.module(
                    conv_input,
                    obs_shape=torch.tensor(conv_input.shape[-3:]).tolist())
        elif isinstance(self.cnn, BetaVAE):
            mu, log_var = self.cnn.encode(conv_input)
            z = self.cnn.reparameterize(mu, log_var)
            cnn_embs = z
        else:
            cnn_embs = self.cnn(
                conv_input, output_stage=self.cnn_output_stage,
                film_inputs=film_inputs)
        h = cnn_embs

        if self.use_spatial_softmax:
            h = self.spatial_softmax(h)

        if len(h.shape) > 2:
            h = torch.flatten(h, start_dim=1)

        if len(self.aux_tasks) > 0:
            input_obs_without_aux_info, aux_task_to_ground_truth_map = (
                self.split_aux_task_input_from_obs(input_obs))
            extra_non_aux_fc_input = input_obs_without_aux_info.narrow(
                start=self.conv_input_length,
                length=self.added_non_aux_fc_input_size,
                dim=1,
            )

        if self.added_non_aux_fc_input_size != 0:
            h = torch.cat((h, extra_non_aux_fc_input), dim=+1)

            if h.shape[0] > 1 and self.batchnorm:
                # Only during training
                h = self.bn_layer(h)

        if (len(self.aux_tasks) > 0 and
                self.aux_to_feed_fc in ["none", "preds"]):
            aux_outputs = self.aux_task_nets_forward(
                h, film_inputs, aux_task_to_ground_truth_map)
        elif self.gripper_policy_arch == "sep_head":
            aux_outputs = {}
        elif output_cnn_embs_in_aux:
            aux_outputs = {
                "cnn_channel_feats": cnn_embs,
                "extra_non_aux_fc_input": extra_non_aux_fc_input,
            }
            if self.output_dist_learning_cnn_embs_in_aux:
                aux_outputs["cnn_emb_for_dist_learning"] = (
                    cnn_emb_for_dist_learning)

        else:
            aux_outputs = None

        if len(self.aux_tasks) > 0:
            if self.aux_to_feed_fc == "preds":
                aux_outputs_list = [
                    aux_outputs['preds'][task] for task in self.aux_tasks]
                h = torch.cat([h] + aux_outputs_list, dim=1)
            elif self.aux_to_feed_fc == "ground_truths":
                aux_outputs_list = [
                    aux_task_to_ground_truth_map[task]
                    for task in self.aux_tasks]
                h = torch.cat([h] + aux_outputs_list, dim=1)

        if self.gripper_policy_arch == "sep_head":
            gripper_fc_out = self.gripper_fc(h)  # (n, 3) or (n, 1)
            gripper_preds = {}
            if self.gripper_loss_type == "ce":
                gripper_preds["gripper_logits"] = gripper_fc_out  # (n, 3)
                gripper_preds["gripper_probs"] = F.softmax(
                    gripper_preds["gripper_logits"], dim=-1)
                gripper_preds["gripper_actions"] = (
                    gripper_preds["gripper_probs"] @
                    ptu.from_numpy(np.array([[-1.], [0.], [1.]])))  # (n, 1)
                # Takes a weighted average of the logits on the
                # interval [-1, 1]
            elif self.gripper_loss_type == "mse":
                gripper_preds["gripper_actions"] = gripper_fc_out  # (n, 1)
            else:
                raise NotImplementedError
            aux_outputs["preds"] = gripper_preds

        h = self.apply_forward_fc(h)
        if output_cnn_embs_in_aux:
            aux_outputs["pre_last_fc_embs"] = h
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)

        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                if output_cnn_embs_in_aux:
                    aux_outputs["log_std_logits"] = self.log_std_logits
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std), stats_dict, aux_outputs

    def apply_forward_fc(self, h):
        # abridged version of CNN.apply_forward_fc
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_last_layer(self):
        self.cnn.unfreeze_last_layer()

    def unfreeze_film_blocks(self):
        self.cnn.unfreeze_film_blocks()

    def freeze_film_blocks(self):
        self.cnn.freeze_film_blocks()

    def unfreeze_fc_layers(self):
        layers_to_unfreeze = (
            [layer for layer in self.fc_layers] + [self.last_fc])
        # seems like there's no trainable params in SpatialSoftmax

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        assert self.std_architecture == "values"
        self.log_std_logits.requires_grad = True

    def init_aux_task_nets(self):
        aux_task_net_input_size = (
            self.cnn_out_dim
            + self.added_non_aux_fc_input_size
            + sum(self.film_emb_dim_list))
        aux_task_to_net_map = {}
        for task in self.aux_tasks:
            exec(f"self.{task}_net = Mlp("
                 f"hidden_sizes=[32], output_size=2, "
                 f"input_size={aux_task_net_input_size}, init_w=1e-3,)")
            aux_task_to_net_map[task] = eval(f"self.{task}_net")
        return aux_task_to_net_map

    def split_aux_task_input_from_obs(self, input_obs):
        aux_task_to_ground_truth_map = {}
        aux_task_idx_intervals = []
        self.verif_thresh = 0.95
        for aux_task in self.aux_tasks:
            start_idx, end_idx = self.obs_key_to_obs_idx_pairs[aux_task]
            aux_task_idx_intervals.append((start_idx, end_idx))
            aux_task_to_ground_truth_map[aux_task] = (
                input_obs[:, start_idx:end_idx])

            # Check that the aux is within the bounds of the observation space
            hi = ptu.from_numpy(self.aux_obs_bounds[aux_task]["hi"])
            lo = ptu.from_numpy(self.aux_obs_bounds[aux_task]["lo"])
            if hi.shape[0] > 10:
                avg_lo_satisfied = float(torch.mean(
                    (lo <= aux_task_to_ground_truth_map[aux_task]).float()))
                avg_hi_satisfied = float(torch.mean(
                    (aux_task_to_ground_truth_map[aux_task] <= hi).float()))
                assert avg_lo_satisfied >= self.verif_thresh, avg_lo_satisfied
                assert avg_hi_satisfied >= self.verif_thresh, avg_hi_satisfied
        sorted_aux_task_idx_intervals = sorted(aux_task_idx_intervals)

        non_aux_obs_list = []
        for i in range(len(sorted_aux_task_idx_intervals) + 1):
            if i == 0:
                start_idx = 0
            if i < len(sorted_aux_task_idx_intervals):
                aux_start_idx, aux_end_idx = sorted_aux_task_idx_intervals[i]
                end_idx = aux_start_idx
            elif i == len(sorted_aux_task_idx_intervals):
                end_idx = input_obs.shape[1]
            non_aux_obs_chunk = input_obs[:, start_idx:end_idx]
            if np.prod(non_aux_obs_chunk.shape) > 0:
                non_aux_obs_list.append(non_aux_obs_chunk)
            start_idx = aux_end_idx
        non_aux_obs = torch.cat(non_aux_obs_list, dim=1)
        return non_aux_obs, aux_task_to_ground_truth_map

    def create_obs_key_to_obs_idx_pair_map(self):
        self.obs_key_to_obs_idx_pairs = {}
        end_idx = None
        for i, obs_key in enumerate(self.observation_keys):
            if i == 0:
                start_idx = 0
            else:
                start_idx = end_idx

            obs_key_dim = self.obs_key_to_dim_map[obs_key]
            if obs_key == "lang_embedding" and self.lang_emb_obs_is_tokens:
                obs_key_dim = LONGEST_SENTENCE_LEN

            end_idx = start_idx + obs_key_dim

            self.obs_key_to_obs_idx_pairs[obs_key] = (start_idx, end_idx)

        print("self.obs_key_to_obs_idx_pairs", self.obs_key_to_obs_idx_pairs)

    def aux_task_nets_forward(
            self, h, film_inputs, aux_task_to_ground_truth_map):
        aux_task_input = torch.cat([h] + film_inputs, dim=1)
        aux_outputs = {"preds": {}, "losses": {}}
        for task, aux_net in self.aux_task_to_net_map.items():
            aux_outputs['preds'][task] = aux_net(aux_task_input)
            aux_task_target = aux_task_to_ground_truth_map[task]
            aux_outputs['losses'][task] = self.aux_task_to_loss_map[task](
                aux_outputs['preds'][task], aux_task_target)
        return aux_outputs
