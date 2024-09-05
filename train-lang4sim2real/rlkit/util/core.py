import numpy as np
import torch

import rlkit.util.pytorch_util as ptu


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    elif isinstance(np_array_or_other, list):
        # This case is only for the list for film_inputs
        return [torch_ify(elem) for elem in np_array_or_other]
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if isinstance(v, dict):
            yield k, v
        elif v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        processed_batch = {}
        for k, x in _filter_batch(np_batch):
            if isinstance(x, dict):
                processed_batch[k] = np_to_pytorch_batch(x)
            elif x.dtype != np.dtype('O'):
                processed_batch[k] = _elem_or_tuple_to_variable(x)
            else:
                pass
        return processed_batch
    else:
        return _elem_or_tuple_to_variable(np_batch)
