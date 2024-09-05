import numpy as np
import torch
from torch import nn


def identity(x):
    return x


_str_to_activation = {
    'identity': identity,
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
}


def activation_from_string(string):
    return _str_to_activation[string]


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_fc_layers(
        hidden_sizes, output_size, fc_normalization_type,
        conv_output_flat_size, added_fc_input_size, init_w):
    fc_layers = nn.ModuleList()
    fc_norm_layers = nn.ModuleList()

    fc_input_size = conv_output_flat_size
    # used only for injecting input directly into fc layers
    fc_input_size += added_fc_input_size
    for idx, hidden_size in enumerate(hidden_sizes):
        fc_layer = nn.Linear(fc_input_size, hidden_size)
        fc_input_size = hidden_size

        fc_layer.weight.data.uniform_(-init_w, init_w)
        fc_layer.bias.data.uniform_(-init_w, init_w)

        fc_layers.append(fc_layer)

        if fc_normalization_type == 'batch':
            fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
        if fc_normalization_type == 'layer':
            fc_norm_layers.append(nn.LayerNorm(hidden_size))

    if output_size is not None:
        last_fc = nn.Linear(fc_input_size, output_size)
        last_fc.weight.data.uniform_(-init_w, init_w)
        last_fc.bias.data.uniform_(-init_w, init_w)
    else:
        last_fc = None

    return fc_layers, fc_norm_layers, last_fc


"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def randint(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randint(*sizes, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
