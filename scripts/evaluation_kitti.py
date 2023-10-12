import open3d as o3d  # prevent loading error
import os
import sys
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


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


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

  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
  data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()

  test_iter = test_loader.__iter__()
  N = len(test_iter)
  n_gpu_failures = 0

  # downsample_voxel_size = 2 * config.voxel_size

  for i in range(len(test_iter)):
    data_timer.tic()
    try:
      data_dict = next(test_iter)
    except ValueError:
      n_gpu_failures += 1
      logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
      continue
    data_timer.toc()
    xyz0, xyz1 = data_dict['raw_pcd_src'], data_dict['raw_pcd_tgt']
    T_gth = data_dict['tsfm']

    xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)

    with torch.no_grad():
      feat_timer.tic()
      sinput_src = data_dict['sinput_src'].to(device)
      # print(sinput_src.F)
      sinput_tgt = data_dict['sinput_tgt'].to(device)

      src_image = data_dict['src_range_image'].to(device)
      tgt_image = data_dict['tgt_range_image'].to(device)

      src_px, src_py= data_dict['src_px'], data_dict['src_py']
      tgt_px, tgt_py= data_dict['tgt_px'], data_dict['tgt_py']

      src_px = [s_x.to(device) for s_x in src_px]
      src_py = [s_y.to(device) for s_y in src_py]
      tgt_px = [t_x.to(device) for t_x in tgt_px]
      tgt_py = [t_y.to(device) for t_y in tgt_py]

      F0 = model(sinput_src,src_image,src_py,src_px).detach()
      F1 = model(sinput_tgt,tgt_image,tgt_py,tgt_px).detach()
      feat_timer.toc()

    feat0 = make_open3d_feature(F0, 32, F0.shape[0])
    feat1 = make_open3d_feature(F1, 32, F1.shape[0])

    reg_timer.tic()
    distance_threshold = config.voxel_size
    # distance_threshold = 0.3

    ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
        pcd0, pcd1, feat0, feat1, distance_threshold*0.8,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    reg_timer.toc()

    # Translation error
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[0, :3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[0, :3, :3]) - 1) / 2)

    # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
    # http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf
    if rte < 2:
      rte_meter.update(rte)

    if not np.isnan(rre) and rre < np.pi / 180 * 5:
      rre_meter.update(rre)

    if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
      success_meter.update(1)
    else:
      success_meter.update(0)
      logging.info(f"Failed with RTE: {rte}, RRE: {rre}")

    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}"
          + f" ({success_meter.avg * 100} %)")
      data_timer.reset()
      feat_timer.reset()
      reg_timer.reset()

  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)")


if __name__ == '__main__':

  dataset_path = "/home/huile/zhangliyuan/Dataset/Kitti"
  output_path = "/home/huile/zhangliyuan/Code/MRFNet/outputs_kitti/exp_pointminkunet_changeencoder"

  checkpoint_path = "/home/huile/zhangliyuan/Code/MRFNet/outputs_kitti/exp_pointminkunet_changeencoder/best_val_checkpoint_epoch_184_rte_0.20181455346522853.pth"


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

