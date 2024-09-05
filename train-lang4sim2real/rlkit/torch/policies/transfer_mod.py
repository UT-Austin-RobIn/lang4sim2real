import torch
from torch import nn

from rlkit.torch.networks.resnet import SpatialSoftmax
from rlkit.torch.policies.base import TorchStochasticPolicy
from rlkit.util.distributions import MultivariateDiagonalNormal
from rlkit.util.pythonplusplus import identity
import rlkit.util.pytorch_util as ptu


class ResidualGaussianPolicyWrapper(TorchStochasticPolicy):
    def __init__(
            self,
            gaussian_policy,
            gaussian_action_dim,
            hidden_sizes,
            obs_state_dim,
            phase3_unfrozen_mods,
            std_architecture="shared",
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            init_w=1e-3,
            use_spatial_softmax=True,
            max_log_std=None,
            min_log_std=None,
            phase3_fn_type="resid_on_action_dist",
            **kwargs):
        """
        Outputs an additive residual correction to a frozen gaussian policy's
        outputs.
        """
        super().__init__(**kwargs)
        self.gaussian_policy = gaussian_policy
        self.fc_normalization_type = "none"
        self.phase3_unfrozen_mods = phase3_unfrozen_mods
        self.std_architecture = std_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_spatial_softmax = use_spatial_softmax
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        self.phase3_fn_type = phase3_fn_type
        self.gripper_policy_arch = "ac_dim"
        self.obs_key_to_obs_idx_pairs = (
            self.gaussian_policy.obs_key_to_obs_idx_pairs)
        self.aux_tasks = self.gaussian_policy.aux_tasks

        if self.phase3_fn_type == "resid_on_pre_last_fc_embs":
            assert (
                hidden_sizes[-1]
                == self.gaussian_policy.fc_layers[-1].out_features)

        if self.phase3_fn_type == "resid_wo_action_dist_inputs":
            self.input_size = (
                self.gaussian_policy.cnn_out_dim +
                obs_state_dim)
        elif self.phase3_fn_type in [
                "non_resid_with_pre_last_fc_embs",
                "resid_on_pre_last_fc_embs"]:
            self.input_size = (
                self.gaussian_policy.cnn_out_dim +
                obs_state_dim +
                self.gaussian_policy.fc_layers[-1].out_features
                # output dim of pre_last fc layer.
            )
        elif self.phase3_fn_type == "resid_on_action_dist":
            self.input_size = (
                self.gaussian_policy.cnn_out_dim +
                (2 * gaussian_action_dim) +
                obs_state_dim
            )
        else:
            raise NotImplementedError

        if self.use_spatial_softmax:
            self.init_spatial_softmax()

        self.fc_layers, self.fc_norm_layers, self.last_fc = (
            ptu.initialize_fc_layers(
                hidden_sizes, gaussian_action_dim,
                self.fc_normalization_type, self.input_size,
                added_fc_input_size=0, init_w=init_w))

        assert self.std_architecture == "values"
        self.log_std_logits = nn.Parameter(
            ptu.zeros(gaussian_action_dim, requires_grad=True))

    def init_spatial_softmax(self):
        test_mat = torch.zeros(
            1,
            self.gaussian_policy.image_size[2],
            self.gaussian_policy.image_size[0],
            self.gaussian_policy.image_size[1],
        ).to(ptu.device)
        test_mat_cnn_kwargs = {}
        if len(self.gaussian_policy.film_emb_dim_list) > 0:
            test_film_embs = [
                torch.zeros(1, film_emb_dim).to(ptu.device)
                for film_emb_dim in self.gaussian_policy.film_emb_dim_list]
            if self.gaussian_policy.film_attn_net is not None:
                test_film_embs, _ = self.gaussian_policy.film_attn_net(
                    torch.zeros(1, self.gaussian_policy.state_obs_dim),
                    test_film_embs)
            test_mat_cnn_kwargs.update(film_inputs=test_film_embs)
        test_mat = self.gaussian_policy.cnn(
            test_mat, output_stage=self.gaussian_policy.cnn_output_stage,
            **test_mat_cnn_kwargs)
        print("test_mat.shape", test_mat.shape)
        self.spatial_softmax = SpatialSoftmax(
            test_mat.shape[2], test_mat.shape[3], test_mat.shape[1])

    def forward(self, obs, film_inputs=[]):
        if len(self.phase3_unfrozen_mods) == 0:
            with torch.no_grad():
                dist, stats_dict, aux_outputs = self.gaussian_policy(
                    obs, film_inputs, output_cnn_embs_in_aux=True)
            # Detaching outputs; we do not backpropogate through frozen policy.
            policy_mean = dist.mean.detach()
            policy_std = dist.stddev.detach()
            obs_embs = aux_outputs['cnn_channel_feats'].detach()
            pre_last_fc_embs = aux_outputs['pre_last_fc_embs'].detach()
            policy_log_std_logits = aux_outputs['log_std_logits'].detach()
        else:
            dist, stats_dict, aux_outputs = self.gaussian_policy(
                obs, film_inputs, output_cnn_embs_in_aux=True)
            policy_mean = dist.mean
            policy_std = dist.stddev
            obs_embs = aux_outputs['cnn_channel_feats']
            pre_last_fc_embs = aux_outputs['pre_last_fc_embs']
            policy_log_std_logits = aux_outputs['log_std_logits']

        # Run cnn_embs through spatial softmax
        if self.use_spatial_softmax:
            obs_embs = self.spatial_softmax(obs_embs)
        else:
            obs_embs = torch.flatten(obs_embs, start_dim=1)

        # Get Obs state
        obs_state = aux_outputs['extra_non_aux_fc_input'].detach()

        if self.phase3_fn_type == "resid_wo_action_dist_inputs":
            x = torch.cat([obs_embs, obs_state], dim=1)
        elif self.phase3_fn_type in [
                "non_resid_with_pre_last_fc_embs",
                "resid_on_pre_last_fc_embs"]:
            x = torch.cat([obs_embs, obs_state, pre_last_fc_embs], dim=1)
        elif self.phase3_fn_type == "resid_on_action_dist":
            x = torch.cat(
                [obs_embs, obs_state, policy_mean, policy_std], dim=1)
        else:
            raise NotImplementedError

        h = self.apply_forward_fc(x)
        if self.phase3_fn_type == "resid_on_pre_last_fc_embs":
            h = h + pre_last_fc_embs 
        preactivation = self.last_fc(h)

        # calc mean and std
        mean_residual = self.output_activation(preactivation)
        if self.phase3_fn_type in [
                "non_resid_with_pre_last_fc_embs",
                "resid_on_pre_last_fc_embs"]:
            mean = mean_residual
            log_std_logits = self.log_std_logits
        elif self.phase3_fn_type in [
                "resid_on_action_dist", "resid_wo_action_dist_inputs"]:
            mean = policy_mean + mean_residual
            log_std_logits = policy_log_std_logits + self.log_std_logits
        else:
            raise NotImplementedError

        log_std = torch.sigmoid(log_std_logits)
        log_std = self.min_log_std + log_std * (
            self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)

        return MultivariateDiagonalNormal(mean, std), stats_dict, aux_outputs

    def apply_forward_fc(self, h):
        # copied from GaussianStandaloneCNNPolicy
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h
