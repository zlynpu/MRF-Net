import time

from models.utils.rangeBrock import *
from models.utils.voxelBlock_fcgf import DownVoxelStage,UpVoxelStage,BasicConvolutionBlock,BasicDeconvolutionBlock,ResidualBlock,UpVoxelStage_withoutres
from lib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import PointTensor, SparseTensor
import torchsparse.nn as spnn

class GFM(nn.Module):
    def __init__(self,in_features):
        super(GFM, self).__init__()

        self.voxel_branch = nn.Linear(in_features,2)
        self.point_branch = nn.Linear(in_features,2)
        # self.range_branch = nn.Linear(in_features,3)


    def forward(self,p,v):


        v2p = voxel_to_point(v, p)
        # print("Layer gfm - NaN check:", torch.isnan(r2p).any())

        p_weight = self.point_branch(p.F)
        v_weight = self.voxel_branch(v2p)
        # print(r_weight.shape,p_weight.shape,v_weight.shape)

        all = p_weight + v_weight
        weight_map = F.softmax(all,dim=-1)
        # print(weight_map.shape)

        p_weight = weight_map[:,0].unsqueeze(1)
        v_weight = weight_map[:,1].unsqueeze(1)
        # print(r_weight.shape,p_weight.shape,v_weight.shape)

        fuse = p.F * p_weight + v2p *v_weight

        p.F = fuse
        v = point_to_voxel(v,p)
        # r = point_to_range(r.shape[-2:],p.F,px,py)
        # print("Layer 0 - NaN check:", torch.isnan(r).any())

        return p,v

class PointMinkUNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(PointMinkUNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 32

        ''' GFM fuse'''
        self.gfm_stem = GFM(self.cs[0])
        self.gfm_stage2 = GFM(self.cs[1])
        self.gfm_stage4 = GFM(self.cs[2])
        self.gfm_stage8 = GFM(self.cs[3])
        self.gfm_decoder = GFM(self.cs[5])
        self.gfm_final = GFM(self.output_channel)

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
        
        ''' point branch '''
        self.point_stem = nn.ModuleList([
            # 32
            nn.Sequential(
                nn.Linear(4,self.cs[0]),
                nn.BatchNorm1d(self.cs[0]),
                nn.ReLU(True),
            ),
            # 64
            nn.Sequential(
                nn.Linear(self.cs[0], self.cs[1]),
                nn.BatchNorm1d(self.cs[1]),
                nn.ReLU(True),
            ),
            # 128
            nn.Sequential(
                nn.Linear(self.cs[1], self.cs[2]),
                nn.BatchNorm1d(self.cs[2]),
                nn.ReLU(True),
            ),
            # 256
            nn.Sequential(
                nn.Linear(self.cs[2], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(self.cs[3], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.ReLU(True),
            ),
            # 32
            nn.Sequential(
                nn.Linear(self.cs[5], self.output_channel),
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(True),
            ),
        ])

        self.final = nn.Linear(self.output_channel,self.output_channel)

        # self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,lidar,image,py,px):
        points = PointTensor(lidar.F,lidar.C.float())
        v0 = initial_voxelize(points,self.vsize)

        v1 = self.voxel_steam(v0)
        points.F = self.point_stem[0](points.F)
        points, v1 = self.gfm_stem(points,v1)
        
        voxel_s1 = self.voxel_block(v1)
        # downsample 2
        voxel_s2 = self.voxel_down1(voxel_s1)
        points.F = self.point_stem[1](points.F)
        points, v2 = self.gfm_stage2(points,voxel_s2)
        # downsample 4
        voxel_s4 = self.voxel_down2(v2)
        points.F = self.point_stem[2](points.F)
        points, v4 = self.gfm_stage4(points,voxel_s4)
        # downsample 8
        voxel_s8 = self.voxel_down3(v4)
        points.F = self.point_stem[3](points.F)
        points, v8 = self.gfm_stage8(points,voxel_s8)
        # upsample
        voxel_s4_tr = self.voxel_up1(v8)
        voxel_s2_tr = self.voxel_up2(voxel_s4_tr, v4)
        points.F = self.point_stem[4](points.F)
        points, voxel_s2_tr = self.gfm_decoder(points,voxel_s2_tr)

        voxel_s1_tr = self.voxel_up3(voxel_s2_tr, v2)
        voxel_out = self.voxel_skip(voxel_s1_tr, voxel_s1)
        # final  decoder fusion
        voxel_out_final = self.voxel_final(voxel_out)
        points.F = self.point_stem[5](points.F)
        points, voxel_out_final = self.gfm_final(points,voxel_out_final)

        out = self.final(points.F)
        out_norm = out / torch.norm(out, p=2, dim=1, keepdim=True)

        return out_norm



