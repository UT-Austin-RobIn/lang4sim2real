import abc

from torch import nn

from rlkit.util.distributions import Distribution


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError
