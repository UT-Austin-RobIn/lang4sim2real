import numpy as np
import torch
import torch.nn as nn

from rlkit.torch.networks.mlp import Mlp


class RandNetAdv(nn.Module):
    """
    Random Network Adversary
    Used as an MLP for automatic domain randomization
    """
    def __init__(self, input_size, action_size, action_space_by_dim):
        super().__init__()
        self.net_out_dims_per_act_dim = 5
        self.action_bins_per_dim = 2**self.net_out_dims_per_act_dim - 1
        self.action_size = action_size
        self.net = Mlp(
            hidden_sizes=[265, 265],
            output_size=self.net_out_dims_per_act_dim * self.action_size,
            # 31 discrete action bins per dim --> represented by 5 dimensions
            input_size=input_size,
        )
        self.net.eval()

        # action_space_by_dim in the format of
        # [[action_dim0_min, action_dim0_max],
        #  [action_dim1_min, action_dim1_max], ...]
        self.action_space_by_dim = np.array(action_space_by_dim)

    def forward(self, x):
        with torch.no_grad():
            out = self.net(x)

        # Get discrete action tokens (from 0-31) per action dim
        action_tokens = []
        for i in range(out.shape[0]):
            action_tokens_by_dim = []
            for j in range(self.action_size):
                start = self.net_out_dims_per_act_dim * j
                end = self.net_out_dims_per_act_dim * (j + 1)
                bitlist = [str(int(x)) for x in out[i][start:end] > 0.0]
                bitstr = "".join(bitlist)
                action_token = eval("0b" + bitstr)
                action_tokens_by_dim.append(action_token)
            action_tokens.append(action_tokens_by_dim)

        # Convert discrete action tokens to action space
        action_tokens = np.array(action_tokens)  # (n, self.action_size)
        act_min = self.action_space_by_dim.T[0]
        act_max = self.action_space_by_dim.T[1]
        actions = (
            (action_tokens / self.action_bins_per_dim) * (act_max - act_min)
            + act_min)
        return actions
