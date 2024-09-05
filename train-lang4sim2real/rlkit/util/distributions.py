"""
Add custom distributions in addition to th existing ones
"""
from collections import OrderedDict

import torch
from torch.distributions import (
    Distribution as TorchDistribution,
    Independent as TorchIndependent,
    Normal as TorchNormal,
)

from rlkit.util.eval_util import create_stats_ordered_dict
import rlkit.util.pytorch_util as ptu


class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


class Delta(Distribution):
    """A deterministic distribution"""
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0

    def __repr__(self):
        return 'Delta({})'.format(self.value)


class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag, validate_args=False),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()
