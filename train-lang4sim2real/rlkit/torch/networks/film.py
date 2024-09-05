import numpy as np
import torch
from torch import nn

from rlkit.torch.networks.mlp import Mlp
from rlkit.util.pythonplusplus import identity
import rlkit.util.pytorch_util as ptu

HIDDEN_ACTIVATION_STR_TO_FN_MAP = {
    "identity": identity,
    "relu": nn.ReLU(inplace=True),
}


class FiLMBlock(nn.Module):
    """FiLM corresponding to one resnet basic block"""
    def __init__(self, emb_dim, num_filters, hidden_sizes=[]):
        assert emb_dim > 0
        super(FiLMBlock, self).__init__()
        self.gamma_net = self.init_single_gamma_or_beta_net(
            emb_dim, num_filters, hidden_sizes)
        self.beta_net = self.init_single_gamma_or_beta_net(
            emb_dim, num_filters, hidden_sizes)

    def init_single_gamma_or_beta_net(
            self, emb_dim, num_filters, hidden_sizes):
        # Old version that worked: single projection layer.
        # net = nn.Linear(emb_dim, num_filters, bias=True)
        # net.weight.data.uniform_(-1e-3, 1e-3)
        # net.bias.data.uniform_(-1e-3, 1e-3)
        net = Mlp(
            hidden_sizes=hidden_sizes,
            output_size=num_filters,
            input_size=emb_dim,
            init_w=1e-3,
            )
        return net

    def forward(self, input_dict):
        # [batch, 2*filters] -> [batch, 1, 1, 2*filters] for broadcasting
        # This above doesn't work in the context of the pytorch resnet.
        # [batch, filters] -> [batch, filters, 1, 1]
        x = input_dict['x']
        film_input = input_dict['film_input']
        assert film_input is not None
        gamma = self.gamma_net(film_input).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_net(film_input).unsqueeze(-1).unsqueeze(-1)
        x = (1 + gamma) * x + beta
        return x


class FiLMBlockSequence(nn.Module):
    """Contains >= 1 FiLM blocks."""
    def __init__(
            self, film_emb_dim_list, num_film_blocks, num_filters,
            hidden_sizes=[], hidden_activation="identity"):
        super(FiLMBlockSequence, self).__init__()
        self.film_emb_dim_list = film_emb_dim_list
        # A list of film emb dim for each block
        self.num_film_blocks = num_film_blocks
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = HIDDEN_ACTIVATION_STR_TO_FN_MAP[
            hidden_activation]
        self.film_blocks = self.init_film_blocks(num_filters)

    def init_film_blocks(self, inplanes):
        """Returns a list of film blocks, one for each input embedding."""
        film_blocks = []
        if len(self.film_emb_dim_list) > 0:
            assert self.num_film_blocks == len(self.film_emb_dim_list)
            for i in range(self.num_film_blocks):
                film_block = FiLMBlock(
                    self.film_emb_dim_list[i], inplanes, self.hidden_sizes)
                film_blocks.append(film_block)
        return nn.Sequential(*film_blocks)

    def forward(self, img_input, film_inputs):
        assert isinstance(film_inputs, list)
        assert len(film_inputs) == len(self.film_blocks)
        out = img_input
        for i, (film_block, film_input) in enumerate(
                zip(self.film_blocks, film_inputs)):
            out = film_block({"x": out, "film_input": film_input})
            if i < len(self.film_blocks) - 1:
                # If not the last layer, apply hidden activation
                out = self.hidden_activation(out)
        return out


class FiLMAttnNet(Mlp):
    def __init__(self, state_obs_dim, num_film_inputs, emb_obs_keys, **kwargs):
        assert len(emb_obs_keys) == num_film_inputs
        super().__init__(
            output_size=num_film_inputs,
            input_size=state_obs_dim, **kwargs)
        self.emb_obs_keys = emb_obs_keys

    def forward(self, input, film_inputs):
        film_attn_weights = super().forward(input)
        if len(film_inputs[0].shape) == 1:
            film_inputs = torch.vstack(film_inputs)
            film_inputs_weighted_combo = film_attn_weights @ film_inputs
        elif len(film_inputs[0].shape) == 2:
            # lists of (B, 768) --> (B, 1, 768)
            film_inputs = [x.unsqueeze(dim=1) for x in film_inputs]
            # lists of (B, 1, 768) --> (B, num_film_inputs, 768)
            film_inputs = torch.cat(film_inputs, dim=1)
            film_inputs_weighted_combo = film_attn_weights @ film_inputs
            film_inputs_weighted_combo = torch.diagonal(
                film_inputs_weighted_combo, dim1=0, dim2=1).T
            # (B, 768)
        else:
            raise NotImplementedError
        stats_dict = self.get_stats_dict(film_attn_weights)
        return [film_inputs_weighted_combo], stats_dict

    def get_stats_dict(self, film_attn_weights):
        stats_dict = {}
        # Calculate avg film attn weight
        film_attn_weights = ptu.get_numpy(film_attn_weights)
        film_attn_avg_weights = np.mean(film_attn_weights, axis=0)
        for i, film_attn_weight in enumerate(film_attn_avg_weights):
            key = f"film_avg_{self.emb_obs_keys[i]}_weight"
            stats_dict[key] = film_attn_weight
        return stats_dict
