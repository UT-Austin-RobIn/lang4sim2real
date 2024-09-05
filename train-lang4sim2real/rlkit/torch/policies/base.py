import abc

import torch

from rlkit.torch.networks.distribution_generator import DistributionGenerator
from rlkit.util.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.util.distributions import Delta


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, **kwargs):
        actions, stats_dict, aux_outputs = self.get_actions(
            obs_np[None], **kwargs)
        return actions[0, :], stats_dict, aux_outputs

    def get_actions(self, obs_np, **kwargs):
        dist, stats_dict, aux_outputs = self._get_dist_from_np(
            obs_np, **kwargs)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions), stats_dict, aux_outputs

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist, stats_dict, aux_outputs = self(*torch_args, **torch_kwargs)
        return dist, stats_dict, aux_outputs


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator
        self.gripper_policy_arch = (
            self._action_distribution_generator.gripper_policy_arch)
        self.gripper_idx = self._action_distribution_generator.gripper_idx

    def forward(self, *args, **kwargs):
        dist, stats_dict, aux_outputs = (
            self._action_distribution_generator.forward(*args, **kwargs))
        action = dist.mle_estimate()

        if self.gripper_policy_arch == "sep_head":
            gripper_action = aux_outputs["preds"]["gripper_actions"]
            action = torch.cat(
                [
                    action[:, :self.gripper_idx],
                    gripper_action,
                    action[:, self.gripper_idx:],
                ],
                dim=1)

        return Delta(action), stats_dict, aux_outputs
