import os
import numpy as np
import argparse
import open3d as o3d
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature

from models import load_model
import torch

def do_voxel_projection(feat,points_xyz,voxel_size):

        feat_one = torch.ones((feat.shape[0],1))
        pc = np.round(points_xyz / voxel_size).astype(np.int32)

        _, inds, inverse_map = sparse_quantize(pc, return_index=True,
                                              return_inverse=True)
        
        coord_,feat_ = (points_xyz[inds],feat[inds])
        # print('coord_:',coord_.shape)
        stensor = SparseTensor(feats=feat_,coords=coord_)
        return stensor, inds

def do_range_projection(points_xyz, points_refl, sel):
        H,W = (64,2048)

        depth = np.linalg.norm(points_xyz,2,axis=1)

        # get scan components
        scan_x = points_xyz[:, 0]
        scan_y = points_xyz[:, 1]
        scan_z = points_xyz[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1

        proj_y = np.cumsum(proj_y)

        proj_x = proj_x * W - 0.001

        # Correct Out-of-Range Indices
        proj_y = np.clip(proj_y, 0, H - 1)  
        proj_x = np.clip(proj_x, 0, W - 1)  

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)

        px = proj_x[sel].copy()
        py = proj_y[sel].copy()
        
        proj_range = np.zeros((H,W)) + 1e-5
        proj_cumsum = np.zeros((H,W)) + 1e-5
        proj_reflectivity = np.zeros((H, W))
        proj_range[proj_y,proj_x] += depth
        proj_cumsum[proj_y,proj_x] += 1
        proj_reflectivity[proj_y, proj_x] += points_refl

        # inverse depth
        proj_range = proj_cumsum / proj_range
        proj_reflectivity = proj_reflectivity / proj_cumsum

        # nomalize values to -10 and 10
        depth_image = 25 * (proj_range - 0.4)
        refl_image = 20 * (proj_reflectivity - 0.5)

        range_image = np.stack([depth_image,refl_image]).astype(np.float32)

        px = px[np.newaxis,:]
        py = py[np.newaxis,:]
        py = 2. * (py / H - 0.5)
        px = 2. * (px / W - 0.5)

        return range_image, px, py

def extract_features(model,
                     xyz,
                     voxel_size=0.05,
                     device=None,
                     ):
    
    

def demo(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_feats = 4  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    Model = load_model(config.model)

    model =Model(
        num_feats,
        vsize=config.voxel_size,
        cr=config.cr,
        cs=config.cs
        )
    checkpoint = torch.load(config.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    pcd = np.fromfile(config.input, dtype=np.float32).reshape(-1, 4)
    xyz_down, feature = extract_features(
      model,
      xyz=pcd[:,:3],
      voxel_size=config.voxel_size,
      device=device,
      skip_check=True)

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)

    vis_pcd = get_colored_point_cloud_feature(vis_pcd,
                                            feature.detach().cpu().numpy(),
                                            config.voxel_size)
    # o3d.visualization.draw_geometries([vis_pcd])
    o3d.io.write_point_cloud('demo.pcd',vis_pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '-i',
      '--input',
      default='/home/huile/zhangliyuan/Dataset/Kitti/dataset/sequences/08/velodyne/000000.bin',
      type=str,
      help='path to a pointcloud file')
    parser.add_argument(
      '-m',
      '--model',
      default='MinkUNet',
      type=str,
      help='path to latest checkpoint (default: None)')
    parser.add_argument(
      '-m',
      '--model',
      default='/home/huile/zhangliyuan/Code/MRFNet/outputs_kitti/exp_minkunet/best_val_checkpoint_epoch_199_rte_0.19064374541630968.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
    parser.add_argument(
      '--voxel_size',
      default=0.3,
      type=float,
      help='voxel size to preprocess point cloud')

    config = parser.parse_args()
    demo(config)