# -*- coding: utf-8 -*-
# ---------------------

import abc
import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def ignore_collate(ignore_keys):

    def collate_fn(batch):
        return _default_collate(batch, ignore_keys)
    return collate_fn

def _default_collate(batch, ignore_keys):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return _default_collate([torch.as_tensor(b) for b in batch], ignore_keys)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: _default_collate([d[key] for d in batch], ignore_keys)
                if key not in ignore_keys else [d[key] for d in batch] for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: _default_collate([d[key] for d in batch], ignore_keys) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_default_collate(samples, ignore_keys) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [_default_collate(samples, ignore_keys) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([_default_collate(samples, ignore_keys) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [_default_collate(samples, ignore_keys) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
