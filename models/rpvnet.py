import time

from models.utils.rangeBrock import *
from models.utils.voxelBlock import DownVoxelStage,UpVoxelStage
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


class RPVnet(nn.Module):
    def __init__(self,vsize=0.05,**kwargs):
        super(RPVnet, self).__init__()

        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        # self.num_classes = kwargs.get('num_classes')
        self.cs = (self.cs*self.cr).int()


        ''' voxel branch '''
        self.voxel_stem = nn.Sequential(
            spnn.Conv3d(1, self.cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(self.cs[0]), spnn.ReLU(True),
            spnn.Conv3d(self.cs[0], self.cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(self.cs[0]), spnn.ReLU(True))
        self.voxel_down1 = DownVoxelStage(self.cs[0],self.cs[1],
                                      b_kernel_size=2,b_stride=2,b_dilation=1,
                                      kernel_size=3,stride=1,dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=2, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down3 = DownVoxelStage(self.cs[2], self.cs[3],
                                      b_kernel_size=2, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down4 = DownVoxelStage(self.cs[3], self.cs[4],
                                      b_kernel_size=2, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_up1 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[3],
                                 b_kernel_size=2,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up2 = UpVoxelStage(self.cs[5],self.cs[6],self.cs[2],
                                 b_kernel_size=2,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up3 = UpVoxelStage(self.cs[6],self.cs[7],self.cs[1],
                                 b_kernel_size=2,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up4 = UpVoxelStage(self.cs[7],self.cs[8],self.cs[0],
                                 b_kernel_size=2,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)

        self.dropout = Block2(dropout_rate=0.3,pooling=False,drop_out=True)

        ''' range branch '''
        self.range_stem = nn.Sequential(
            ResContextBlock(2,self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            # Block1Res(cs[0],cs[0])
        )
        # nn.GatFusionModule

        self.range_stage1 = nn.Sequential(
            Block1Res(self.cs[0],self.cs[1]),
            Block2(dropout_rate=0.2,pooling=True,drop_out=False)
        )
        self.range_stage2 = nn.Sequential(
            Block1Res(self.cs[1],self.cs[2]),
            Block2(dropout_rate=0.2, pooling=True)
        )
        self.range_stage3 = nn.Sequential(
            Block1Res(self.cs[2],self.cs[3]),
            Block2(dropout_rate=0.2, pooling=True)
        )

        self.range_stage4 = nn.Sequential(
            Block1Res(self.cs[3],self.cs[4]),
            Block2(dropout_rate=0.2, pooling=True)
        )
        # nn.GatFusionModule

        self.range_stage5 = Block4(self.cs[4],self.cs[5],self.cs[3],upscale_factor=2,dropout_rate=0.2)
        self.range_stage6 = Block4(self.cs[5],self.cs[6],self.cs[2],2,0.2)
        # nn.GatFusionModule

        self.range_stage7 = Block4(self.cs[6],self.cs[7],self.cs[1],2,0.2)
        self.range_stage8 = Block4(self.cs[7],self.cs[8],self.cs[0],2,0.2,drop_out=False)
        # nn.GatFusionModule

        ''' point branch '''
        self.point_stem = nn.ModuleList([
            # 32
            nn.Sequential(
                nn.Linear(1,self.cs[0]),
                nn.BatchNorm1d(self.cs[0]),
                nn.ReLU(True),
            ),
            # 256
            nn.Sequential(
                nn.Linear(self.cs[0], self.cs[4]),
                nn.BatchNorm1d(self.cs[4]),
                nn.ReLU(True),
            ),
            # 128
            nn.Sequential(
                nn.Linear(self.cs[4], self.cs[6]),
                nn.BatchNorm1d(self.cs[6]),
                nn.ReLU(True),
            ),
            # 32
            nn.Sequential(
                nn.Linear(self.cs[6], self.cs[8]),
                nn.BatchNorm1d(self.cs[8]),
                nn.ReLU(True),
            ),

        ])

        self.gfm_stem = GFM(self.cs[0])
        self.gfm_stage4 = GFM(self.cs[4])
        self.gfm_stage6 = GFM(self.cs[6])
        self.gfm_stage8 = GFM(self.cs[8])

        self.final = nn.Linear(self.cs[8],self.cs[8])

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,lidar,image,py,px):

        points = PointTensor(lidar.F,lidar.C.float())
        # print('point',points.C)
        v0 = initial_voxelize(points,self.vsize)
        if torch.isnan(image).any():
            print('there is nan at first')
        # print('voxel',v0.C)

        ''' Fuse 1 '''
        v0 = self.voxel_stem(v0)
        points.F = self.point_stem[0](points.F) # 32
        range0 = self.range_stem(image) # n,32,64,2048
        # print("Layer 0 - NaN check:", torch.isnan(range0).any())
        # print(range0.shape)
        # print("Layer 1 - NaN check:", torch.isnan(v.F).any())
        range0,points,v0 = self.gfm_stem(range0,points,v0,px,py)
        if torch.isnan(range0).any():
            print('there is nan after gfm')
        # print("Layer 0 - NaN check:", torch.isnan(range0).any())
        # todo: add dropout here?
        # v0.F = self.dropout(v0.F)

        ''' Fuse 2 '''
        # print(v0.F.shape)
        v1 = self.voxel_down1(v0) #64
        v2 = self.voxel_down2(v1) #128
        v3 = self.voxel_down3(v2) #256
        v4 = self.voxel_down4(v3) #256
        # print("Layer 1 - NaN check:", torch.isnan(range0).any())

        points.F = self.point_stem[1](points.F)# 64

        range1 = self.range_stage1(range0) # n,64,32,1024
        range2 = self.range_stage2(range1) # n,128,16,512
        range3 = self.range_stage3(range2) # n,256,8,256
        range4 = self.range_stage4(range3) # n,256,4,128

        range4,points,v4 = self.gfm_stage4(range4,points,v4,px,py)
        v4.F = self.dropout(v4.F)
        # print("Layer 2 - NaN check:", torch.isnan(range4).any())

        ''' Fuse 3 '''
        v5 = self.voxel_up1(v4,v3)
        v6 = self.voxel_up2(v5,v2)

        points.F = self.point_stem[2](points.F)

        range5 = self.range_stage5(range4,range3)
        range6 = self.range_stage6(range5,range2)
        # print("Layer 2.1 - NaN check:", torch.isnan(range6).any())

        range6,points,v6 = self.gfm_stage6(range6,points,v6,px,py)
        # print("Layer 2.2 - NaN check:", torch.isnan(range6).any())
        v6.F = self.dropout(v6.F)

        ''' Fuse 4 '''
        v7 = self.voxel_up3(v6,v1)
        v8 = self.voxel_up4(v7,v0)

        points.F = self.point_stem[3](points.F)

        range7 = self.range_stage7(range6,range1)
        range8 = self.range_stage8(range7,range0)
        # print("Layer 2.5 - NaN check:", torch.isnan(range8).any())
        range8,points,v8 = self.gfm_stage8(range8,points,v8,px,py)

        out = self.final(points.F)
        # print("Layer 3 - NaN check:", torch.isnan(range8).any())
        if torch.isnan(out).any():
            print('there is nan in the end!!')
        return out / torch.norm(out, p=2, dim=1, keepdim=True)
