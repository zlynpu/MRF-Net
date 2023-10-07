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

class MrfNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(MrfNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 32

        ''' GFM fuse'''
        self.gfm_stem = GFM(self.cs[0])
        self.gfm_stage4 = GFM(self.cs[3])
        self.gfm_stage6 = GFM(self.cs[5])
        self.gfm_stage8 = GFM(self.output_channel)

        ''' voxel branch '''
        self.voxel_steam = BasicConvolutionBlock(self.input_channel,self.cs[0],kernel_size=5,
                             stride=1,dilation=1)
        self.voxel_down1 = ResidualBlock(self.cs[0], self.cs[0], kernel_size=3, stride=1, dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[0], self.cs[1],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down3 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down4 = DownVoxelStage(self.cs[2], self.cs[3],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_bottle = nn.Sequential(
            BasicDeconvolutionBlock(self.cs[3], self.cs[4],
                                    kernel_size=3, stride=2),
            ResidualBlock(self.cs[4], self.cs[4],
                          kernel_size=3, stride=1)
        )
        self.voxel_up1 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[2],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up2 = UpVoxelStage(self.cs[5],self.cs[6],self.cs[1],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up3 = UpVoxelStage_withoutres(self.cs[6],self.cs[7],self.cs[0],
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
            # 256
            nn.Sequential(
                nn.Linear(self.cs[0], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(True),
            ),
            # 64
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
        ''' range branch '''
        self.range_stem = nn.Sequential(
            ResContextBlock(2,self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            # Block1Res(cs[0],cs[0])
        )
        # nn.GatFusionModule

        self.range_down1 = nn.Sequential(
            Block1Res(self.cs[0],self.cs[1]),
            Block2(dropout_rate=0.2,pooling=True,drop_out=False)
        )
        self.range_down2 = nn.Sequential(
            Block1Res(self.cs[1],self.cs[2]),
            Block2(dropout_rate=0.2, pooling=True)
        )
        self.range_down3 = nn.Sequential(
            Block1Res(self.cs[2],self.cs[3]),
            Block2(dropout_rate=0.2, pooling=True)
        )

        # nn.GatFusionModule

        self.range_up1 = Block_withoutskip(self.cs[3],self.cs[4],upscale_factor=2,dropout_rate=0.2)
        self.range_up2 = Block4(self.cs[4],self.cs[5],self.cs[2],2,0.2)
        # nn.GatFusionModule

        self.range_up3 = Block4(self.cs[5],self.cs[6],self.cs[1],2,0.2)
        self.range_up4 = Block4(self.cs[6],self.cs[7],self.cs[0],1,0.2,drop_out=False)
        self.range_final = nn.Sequential(
            ResContextBlock(self.cs[7],self.output_channel),
            ResContextBlock(self.output_channel, self.output_channel),
            ResContextBlock(self.output_channel, self.output_channel),
            # Block1Res(cs[0],cs[0])
        )

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
        range0 = self.range_stem(image)
        points.F = self.point_stem[0](points.F)
        range0, points, v1 = self.gfm_stem(range0,points,v1,px,py)
        
        voxel_s1 = self.voxel_down1(v1)
        voxel_s2 = self.voxel_down2(voxel_s1)
        voxel_s4 = self.voxel_down3(voxel_s2)
        voxel_s8 = self.voxel_down4(voxel_s4)

        range_s2 = self.range_down1(range0) # n,64,32,1024
        range_s4 = self.range_down2(range_s2) # n,128,16,512
        
        range_s8 = self.range_down3(range_s4) # n,256,8,256
        
        points.F = self.point_stem[1](points.F)
        range_s8, points, voxel_s8 = self.gfm_stage4(range_s8,points,voxel_s8,px,py)

        voxel_s4_tr = self.voxel_bottle(voxel_s8)
        voxel_s2_tr = self.voxel_up1(voxel_s4_tr, voxel_s4)

        range_s4_tr = self.range_up1(range_s8)
        # print(range_s4_tr.shape, range_s4.shape)
        range_s2_tr = self.range_up2(range_s4_tr, range_s4)

        points.F = self.point_stem[2](points.F)
        range_s2_tr, points, voxel_s2_tr = self.gfm_stage6(range_s2_tr,points,voxel_s2_tr,px,py)

        # voxel_s2_tr.F = self.dropout(voxel_s2_tr.F)
        voxel_s1_tr = self.voxel_up2(voxel_s2_tr, voxel_s2)
        voxel_out = self.voxel_up3(voxel_s1_tr, voxel_s1)
        voxel_out_final = self.voxel_final(voxel_out)

        range_s1_tr = self.range_up3(range_s2_tr, range_s2)
        range_out = self.range_up4(range_s1_tr, range0)
        range_out_final = self.range_final(range_out)
        points.F = self.point_stem[3](points.F)
        range_out_final, points, voxel_out_final = self.gfm_stage8(range_out_final,points,voxel_out_final,px,py)

        out = self.final(points.F)
        out_norm = out / torch.norm(out, p=2, dim=1, keepdim=True)

        return out_norm



