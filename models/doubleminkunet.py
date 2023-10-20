import time

from models.utils.rangeBrock import *
from models.utils.voxelBlock_fcgf import DownVoxelStage,UpVoxelStage,BasicDeconvolutionBlock,ResidualBlock,UpVoxelStage_withoutres,BasicConvolutionBlock
from lib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import PointTensor, SparseTensor
import torchsparse.nn as spnn

class DoubleMinkUNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(DoubleMinkUNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 32

        ''' voxel branch '''
        self.voxel_steam = BasicConvolutionBlock(self.input_channel,self.cs[0],kernel_size=5,
                             stride=1,dilation=1)
        self.voxel_block = ResidualBlock(self.cs[0], self.cs[0], kernel_size=3, stride=1, dilation=1)
        # self.voxel_down1 = ResidualBlock(self.cs[0], self.cs[0], kernel_size=3, stride=1, dilation=1)
        self.voxel_down1 = DownVoxelStage(self.cs[0], self.cs[1],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down3 = DownVoxelStage(self.cs[2], self.cs[3],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_up1 = nn.Sequential(
            BasicDeconvolutionBlock(self.cs[3], self.cs[4],
                                    kernel_size=3, stride=2),
            ResidualBlock(self.cs[4], self.cs[4],
                          kernel_size=3, stride=1)
        )
        self.voxel_up2 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[2],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up3 = UpVoxelStage(self.cs[5],self.cs[6],self.cs[1],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_skip = UpVoxelStage_withoutres(self.cs[6],self.cs[7],self.cs[0],
                                                 kernel_size=3,stride=1)
        self.voxel_final = spnn.Conv3d(self.cs[7],self.output_channel,
                                       kernel_size=1,stride=1,bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def forward(self,lidar_src,image_src,py_src,px_src,lidar_tgt,image_tgt,py_tgt,px_tgt):
        ''' initial voxel ''' 
        point_src = PointTensor(lidar_src.F,lidar_src.C.float())
        point_tgt = PointTensor(lidar_tgt.F,lidar_tgt.C.float())
        v0_src = initial_voxelize(point_src,self.vsize)
        v0_tgt = initial_voxelize(point_tgt,self.vsize)

        ''' downsample '''
        # source point cloud
        v1_src = self.voxel_steam(v0_src)
        voxel_s1_src = self.voxel_block(v1_src)
        voxel_s2_src = self.voxel_down1(voxel_s1_src)
        voxel_s4_src = self.voxel_down2(voxel_s2_src)
        voxel_s8_src = self.voxel_down3(voxel_s4_src)
        # target point cloud
        v1_tgt = self.voxel_steam(v0_tgt)
        voxel_s1_tgt = self.voxel_block(v1_tgt)
        voxel_s2_tgt = self.voxel_down1(voxel_s1_tgt)
        voxel_s4_tgt = self.voxel_down2(voxel_s2_tgt)
        voxel_s8_tgt = self.voxel_down3(voxel_s4_tgt)

        ''' upsample '''
        # source point cloud
        voxel_s4_tr_src = self.voxel_up1(voxel_s8_src)
        voxel_s2_tr_src = self.voxel_up2(voxel_s4_tr_src, voxel_s4_src)
        voxel_s1_tr_src = self.voxel_up3(voxel_s2_tr_src, voxel_s2_src)
        voxel_out_src = self.voxel_skip(voxel_s1_tr_src, voxel_s1_src)
        voxel_out_final_src = self.voxel_final(voxel_out_src)
        # target point cloud
        voxel_s4_tr_tgt = self.voxel_up1(voxel_s8_tgt)
        voxel_s2_tr_tgt = self.voxel_up2(voxel_s4_tr_tgt, voxel_s4_tgt)
        voxel_s1_tr_tgt = self.voxel_up3(voxel_s2_tr_tgt, voxel_s2_tgt)
        voxel_out_tgt = self.voxel_skip(voxel_s1_tr_tgt, voxel_s1_tgt)
        voxel_out_final_tgt = self.voxel_final(voxel_out_tgt)

        ''' project to point '''
        out_src = voxel_to_point(voxel_out_final_src, point_src)
        out_src = out_src / (torch.norm(out_src, p=2, dim=1, keepdim=True))
        out_tgt = voxel_to_point(voxel_out_final_tgt, point_tgt)
        out_tgt = out_tgt / (torch.norm(out_tgt, p=2, dim=1, keepdim=True))

        return out_src, out_tgt





