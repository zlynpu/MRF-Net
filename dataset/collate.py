from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate

__all__ = ['sparse_collate_fn']


def sparse_collate_fn(inputs: List[Any]) -> Any:
    # print(inputs[0].type)
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])

            # px and py are treated separately because their sizes change dynamically
            # todo: Added to SparseTensor to handle variable scale problems, 
            # but requires the last dimension to be batch, which makes indexing slowe
            elif name in ['src_px','src_py','tgt_px','tgt_py']:
                output[name] = [torch.from_numpy(input[name]) for input in inputs]

            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name])for input in inputs], dim=0)

            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs],
                                           dim=0)

            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
            
        return output
    else:
        return inputs