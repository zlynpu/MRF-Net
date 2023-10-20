import time

from models.utils.rangeBrock import *
from models.utils.voxelBlock_rpv import DownVoxelStage,UpVoxelStage
from lib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import PointTensor
import torchsparse.nn as spnn


class GFM(nn.Module):
    def __init__(self,in_features):
        super(GFM, self).__init__()

        self.voxel_branch = nn.Linear(in_features,3)
        self.point_branch = nn.Linear(in_features,3)
        self.range_branch = nn.Linear(in_features,3)


    def forward(self,r,p,v,px,py):


        v2p = voxel_to_point(v, p)
        r2p = range_to_point(r, px, py)
        # print("Layer gfm - NaN check:", torch.isnan(r2p).any())

        r_weight = self.range_branch(r2p)
        p_weight = self.point_branch(p.F)
        v_weight = self.voxel_branch(v2p)
        # print(r_weight.shape,p_weight.shape,v_weight.shape)

        all = r_weight + p_weight + v_weight
        weight_map = F.softmax(all,dim=-1)
        # print(weight_map.shape)

        r_weight = weight_map[:,0].unsqueeze(1)
        p_weight = weight_map[:,1].unsqueeze(1)
        v_weight = weight_map[:,2].unsqueeze(1)
        # print(r_weight.shape,p_weight.shape,v_weight.shape)

        fuse = r2p * r_weight + p.F * p_weight + v2p *v_weight

        p.F = fuse
        v = point_to_voxel(v,p)
        r = point_to_range(r.shape[-2:],p.F,px,py)
        # print("Layer 0 - NaN check:", torch.isnan(r).any())

        return r,p,v


class RPVNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(RPVNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        # self.num_classes = kwargs.get('num_classes')
        self.cs = (self.cs*self.cr).int()

        ''' voxel branch '''
        self.voxel_stem = nn.Sequential(
            spnn.Conv3d(self.input_channel, self.cs[0], kernel_size=5, stride=1),
            spnn.BatchNorm(self.cs[0]), spnn.ReLU(True),
            spnn.Conv3d(self.cs[0], self.cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(self.cs[0]), spnn.ReLU(True))
        self.voxel_down1 = DownVoxelStage(self.cs[0],self.cs[1],
                                      b_kernel_size=3,b_stride=2,b_dilation=1,
                                      kernel_size=3,stride=1,dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down3 = DownVoxelStage(self.cs[2], self.cs[3],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_up1 = UpVoxelStage(self.cs[3],self.cs[4],self.cs[2],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up2 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[1],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up3 = UpVoxelStage(self.cs[5],self.cs[6],self.cs[0],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        
        ''' range branch '''
        self.range_stem = nn.Sequential(
            ResContextBlock(2,self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
        )
        self.range_down1 = Block1Res(self.cs[0], self.cs[1], 0.2, pooling=True, drop_out=False)
        self.range_down2 = Block1Res(self.cs[1], self.cs[2], 0.2, pooling=True, drop_out=False)
        self.range_down3 = Block1Res(self.cs[2], self.cs[3], 0.2, pooling=True, drop_out=False)

        self.range_up1 = UpBlock(self.cs[3],self.cs[4],0.2,drop_out=False,mid_filters=self.cs[3] // 4 + self.cs[2])
        self.range_up2 = UpBlock(self.cs[4],self.cs[5],0.2,drop_out=False,mid_filters=self.cs[4] // 4 + self.cs[1])
        self.range_up3 = UpBlock(self.cs[5],self.cs[6],0.2,drop_out=False,mid_filters=self.cs[5] // 4 + self.cs[0])
 
        ''' point branch '''
        self.point_stem = nn.ModuleList([
            # 32
            nn.Sequential(
                nn.Linear(self.input_channel,self.cs[0]),
                nn.BatchNorm1d(self.cs[0]),
                nn.ReLU(True),
            ),
            # 256
            nn.Sequential(
                nn.Linear(self.cs[0], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(True),
            ),
            # 128
            nn.Sequential(
                nn.Linear(self.cs[3], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.ReLU(True),
            ),
            # 32
            nn.Sequential(
                nn.Linear(self.cs[5], self.cs[6]),
                nn.BatchNorm1d(self.cs[6]),
                nn.ReLU(True),
            ),

        ])

        self.gfm_stem = GFM(self.cs[0])
        self.gfm_stage1 = GFM(self.cs[3])
        self.gfm_stage2 = GFM(self.cs[5])
        self.gfm_stage3 = GFM(self.cs[6])

        self.final = nn.Linear(self.cs[6],self.cs[6])

    def forward(self,lidar,image,py,px):

        points = PointTensor(lidar.F,lidar.C.float())
        v0 = initial_voxelize(points,self.vsize)
        if torch.isnan(image).any():
            print('there is nan at first')
        # print('voxel',v0.C)

        ''' Fuse 1 '''
        v1 = self.voxel_stem(v0)
        points.F = self.point_stem[0](points.F) # 32
        range1 = self.range_stem(image) # n,32,64,2048
        range1,points,v1 = self.gfm_stem(range1,points,v1,px,py)
        if torch.isnan(range1).any():
            print('there is nan after gfm')
        ''' Fuse 2 '''
        v2 = self.voxel_down1(v1) #64
        v4 = self.voxel_down2(v2) #128
        v8 = self.voxel_down3(v4) #256
        
        points.F = self.point_stem[1](points.F)# 64

        range2 = self.range_down1(range1) # n,64,32,1024
        range4 = self.range_down2(range2) # n,128,16,512
        range8 = self.range_down3(range4) # n,256,8,256
        
        range8,points,v8 = self.gfm_stage1(range8,points,v8,px,py)

        ''' Fuse 3 '''
        v4_tr = self.voxel_up1(v8,v4)
        v2_tr = self.voxel_up2(v4_tr,v2)

        points.F = self.point_stem[2](points.F)

        range4_tr = self.range_up1(range8,range4)
        range2_tr = self.range_up2(range4_tr,range2)
        
        range2_tr,points,v2_tr = self.gfm_stage2(range2_tr,points,v2_tr,px,py)

        ''' Fuse 4 '''
        v1_tr = self.voxel_up3(v2_tr,v1)
        
        points.F = self.point_stem[3](points.F)

        range1_tr = self.range_up3(range2_tr,range1)
        
        range1_tr,points,v1_tr = self.gfm_stage3(range1_tr,points,v1_tr,px,py)

        out = self.final(points.F)
        
        if torch.isnan(out).any():
            print('there is nan in the end!!')

        out_norm = out / torch.norm(out, p=2, dim=1, keepdim=True)

        if torch.isnan(out_norm).any():
            print('there is nan in the end!!')
        return out_norm