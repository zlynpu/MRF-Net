import time

from models.utils.rangeBrock import *
from models.utils.voxelBlock_rpv import DownVoxelStage,UpVoxelStage
from lib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import PointTensor
import torchsparse.nn as spnn

class SegMinkUNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(SegMinkUNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 32

        ''' voxel branch '''
        self.voxel_stem = nn.Sequential(
            spnn.Conv3d(self.input_channel, self.cs[0], kernel_size=3, stride=1),
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

        # self.dropout = Block2(dropout_rate=0.3,pooling=False,drop_out=True)

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
        
        v0 = self.voxel_stem(v0)
        v1 = self.voxel_down1(v0) #64
        v2 = self.voxel_down2(v1) #128
        v3 = self.voxel_down3(v2) #256
        v4 = self.voxel_down4(v3) #256

        v5 = self.voxel_up1(v4,v3)
        v6 = self.voxel_up2(v5,v2)
        v7 = self.voxel_up3(v6,v1)
        v8 = self.voxel_up4(v7,v0)

        out = voxel_to_point(v8, points)

        out_norm = out / torch.norm(out, p=2, dim=1, keepdim=True)

        return out_norm
