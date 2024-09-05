"""
General networks for pytorch.
"""
from rlkit.torch.networks.basic import Clamp
from rlkit.torch.networks.cnn import CNN
from rlkit.torch.networks.mlp import Mlp

__all__ = [
    'Clamp',
    'CNN',
    'Mlp'
]
