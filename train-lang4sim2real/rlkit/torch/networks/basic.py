import torch
from torch import nn


class Clamp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__name__ = "Clamp"

    def forward(self, x):
        return torch.clamp(x, **self.kwargs)


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input
