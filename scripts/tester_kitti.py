import open3d as o3d  # prevent loading error
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append(os.path.abspath("/home/huile/zhangliyuan/Code/MRFNet"))
import warnings
warnings.filterwarnings("ignore")

import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from models import load_model

from dataset.dataloader import make_data_loader
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer
from tqdm import tqdm

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return: 
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2) 
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if(isinstance(pcd,torch.Tensor)):
        n1 = pcd.size(0)
    elif(isinstance(pcd, np.ndarray)):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]

def main(config,checkpoint):
    test_loader = make_data_loader(
        config,
        config.test_phase,
        1,
        num_threads=config.test_num_thread,
        shuffle=True
    )

    num_feats = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = load_model(config.model)
    model =Model(
        num_feats,
        vsize=config.voxel_size,
        cr=config.cr,
        cs=config.cs
        )
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model = model.to(device)
    model.eval()
    test_iter = test_loader.__iter__()
    N = int(len(test_loader.dataset))
    tsfm_gt = []
    tsfm_est = []
    with torch.no_grad():
        for _ in tqdm(range(len(test_iter))):
            data_dict = next(test_iter)          
            sinput_src = data_dict['sinput_src'].to(device)
            # print(sinput_src.F)
            sinput_tgt = data_dict['sinput_tgt'].to(device)

            src_image = data_dict['src_range_image'].to(device)
            tgt_image = data_dict['tgt_range_image'].to(device)

            src_px, src_py= [item.to(device) for item in data_dict['src_px']], [item.to(device) for item in data_dict['src_py']]
            tgt_px, tgt_py= [item.to(device) for item in data_dict['tgt_px']], [item.to(device) for item in data_dict['tgt_py']]
            F0 = model(sinput_src,src_image,src_py,src_px).detach()
            F1 = model(sinput_tgt,tgt_image,tgt_py,tgt_px).detach()

            xyz0, xyz1 = data_dict['raw_pcd_src'], data_dict['raw_pcd_tgt']
            T_gth = data_dict['tsfm']
            # print(T_gth[0,:,:].numpy().shape)
            tsfm_gt.append(T_gth[0,:,:].numpy())

            xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

            # n_points = 5000
            # ########################################
            # # run random sampling or probabilistic sampling
            # xyz0np, F0 = random_sample(xyz0np, F0, n_points)
            # xyz1np, F1 = random_sample(xyz1np, F1, n_points)

            pcd0 = make_open3d_point_cloud(xyz0np)
            pcd1 = make_open3d_point_cloud(xyz1np)

            feat0 = make_open3d_feature(F0, 32, F0.shape[0])
            feat1 = make_open3d_feature(F1, 32, F1.shape[0])

            distance_threshold = 0.3
            ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
                pcd0, pcd1, feat0, feat1, distance_threshold*0.8,
                o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.registration.RANSACConvergenceCriteria(50000, 1000))
            T_ransac = ransac_result.transformation
            
            tsfm_est.append(T_ransac)
    tsfm_est = np.array(tsfm_est)
    rot_est = tsfm_est[:,:3,:3]
    trans_est = tsfm_est[:,:3,3]
    tsfm_gt = np.array(tsfm_gt)
    rot_gt = tsfm_gt[:,:3,:3]
    trans_gt = tsfm_gt[:,:3,3]

    rot_threshold = 5
    trans_threshold = 2

    r_deviation = get_angle_deviation(rot_est, rot_gt)
    translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

    flag_1=r_deviation<rot_threshold
    flag_2=translation_errors<trans_threshold
    correct=(flag_1 & flag_2).sum()
    precision=correct/rot_gt.shape[0]

    message=f'\n Registration recall: {precision:.3f}\n'

    r_deviation = r_deviation[flag_1]
    translation_errors = translation_errors[flag_2]

    errors=dict()
    errors['rot_mean']=round(np.mean(r_deviation),3)
    errors['rot_median']=round(np.median(r_deviation),3)
    errors['trans_rmse'] = round(np.mean(translation_errors),3)
    errors['trans_rmedse']=round(np.median(translation_errors),3)
    errors['rot_std'] = round(np.std(r_deviation),3)
    errors['trans_std']= round(np.std(translation_errors),3)

    message+=str(errors)
    print(message)

if __name__ == '__main__':

  dataset_path = "/home/huile/zhangliyuan/Dataset/Kitti"
  output_path = "/home/huile/zhangliyuan/Code/MRFNet/outputs_kitti/exp_mrfnet_withoutdropout"

  checkpoint_path = "/home/huile/zhangliyuan/Code/MRFNet/outputs_kitti/exp_mrfnet_withoutdropout/checkpoint_epoch_103_rte_0.3358605405013077.pth"


  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', default=output_path, type=str)
  parser.add_argument('--test_phase', default='test', type=str)
  parser.add_argument('--test_num_thread', default=8, type=int)
  parser.add_argument('--model', default=checkpoint_path, type=str)

  parser.add_argument('--kitti_root', type=str, default=dataset_path)
  args = parser.parse_args()

  config = json.load(open(args.save_dir + '/config.json', 'r'))
  # config=checkpoint['config']
  config = edict(config)
  config.save_dir = args.save_dir
  config.test_phase = args.test_phase
  config.kitti_root = args.kitti_root
  config.kitti_odometry_root = args.kitti_root + '/dataset'
  config.test_num_thread = args.test_num_thread

  main(config,checkpoint=args.model)
        