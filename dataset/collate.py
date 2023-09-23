from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate

__all__ = ['sparse_collate_fn']


def sparse_collate_fn(inputs):
    input_data = list(inputs)
    # print(input_data)
    output = {}
    output['sinput_src'] = sparse_collate([input['sinput_src'] for input in input_data])
    # print(output['sinput_src'].C.shape)
    output['sinput_tgt'] = sparse_collate([input['sinput_tgt'] for input in input_data])

    output['src_px'] = [torch.from_numpy(input['src_px']).float() for input in input_data]
    # print(len(output['src_px']))
    output['src_py'] = [torch.from_numpy(input['src_py']).float() for input in input_data]
    output['tgt_px'] = [torch.from_numpy(input['tgt_px']).float() for input in input_data]
    output['tgt_py'] = [torch.from_numpy(input['tgt_py']).float() for input in input_data]

    output['scale'] = input_data[0]['scale']
    
    matching_inds_batch = []
    sel0_batch = []
    sel1_batch = []
    tsfm_batch = []
    src_range_image_batch = []
    tgt_range_image_batch = []
    len_batch = []
    curr_start_ind = torch.zeros((1,2))
    curr_start_ind_sel = torch.zeros((1,2))

    for batch_id, _ in enumerate(input_data):
        N0 = input_data[batch_id]['sel0'].shape[0]
        N1 = input_data[batch_id]['sel1'].shape[0]

        N0_sel = input_data[batch_id]['sinput_src'].C.shape[0]
        N1_sel = input_data[batch_id]['sinput_tgt'].C.shape[0] 

        sel0_batch.append(torch.from_numpy(input_data[batch_id]['sel0'])+curr_start_ind_sel[0,0])
        sel1_batch.append(torch.from_numpy(input_data[batch_id]['sel1'])+curr_start_ind_sel[0,1])
        
        matching_inds_batch.append(input_data[batch_id]['correspondence']+curr_start_ind)
        len_batch.append([N0, N1])
        tsfm_batch.append(input_data[batch_id]['tsfm'])

        src_range_image_batch.append(torch.from_numpy(input_data[batch_id]['src_range_image']))
        tgt_range_image_batch.append(torch.from_numpy(input_data[batch_id]['tgt_range_image']))

        curr_start_ind[0,0]+=N0
        curr_start_ind[0,1]+=N1

        curr_start_ind_sel[0,0]+=N0_sel
        curr_start_ind_sel[0,1]+=N1_sel

    output['len_batch'] = len_batch
    output['correspondence'] = torch.cat(matching_inds_batch,0).int()
    # print(output['correspondence'].shape)
    output['sel0'] = torch.cat(sel0_batch,0).int()
    # print(output['sel0'].shape)
    output['sel1'] = torch.cat(sel1_batch,0).int()
    output['src_range_image'] = torch.stack(src_range_image_batch,0).float()
    # print(output['src_range_image'].shape)
    output['tgt_range_image'] = torch.stack(tgt_range_image_batch,0).float()
    output['tsfm'] = torch.stack(tsfm_batch,0).float()
    
    return output
    # print(inputs[0].type)
    # if isinstance(inputs[0], dict):
    #     output = {}
    #     for name in inputs[0].keys():
    #         if isinstance(inputs[0][name], dict):
    #             output[name] = sparse_collate_fn(
    #                 [input[name] for input in inputs])

    #         # px and py are treated separately because their sizes change dynamically
    #         # todo: Added to SparseTensor to handle variable scale problems, 
    #         # but requires the last dimension to be batch, which makes indexing slowe
    #         elif name in ['src_px','src_py','tgt_px','tgt_py']:
    #             output[name] = [torch.from_numpy(input[name]) for input in inputs]
            
    #         elif isinstance(inputs[0][name], np.ndarray):
    #             output[name] = torch.stack(
    #                 [torch.tensor(input[name])for input in inputs], dim=0)

    #         elif isinstance(inputs[0][name], torch.Tensor):
    #             output[name] = torch.stack([input[name] for input in inputs],
    #                                        dim=0)

    #         elif isinstance(inputs[0][name], SparseTensor):
    #             output[name] = sparse_collate([input[name] for input in inputs])
    #         else:
    #             output[name] = [input[name] for input in inputs]
            
    #     return output
    # else:
    #     return inputs