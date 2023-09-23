# coding = utf-8
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models.rpvnet import RPVnet 
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash


class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    # Model = load_model(config.model)

    model = RPVnet(
        vsize=config.voxel_size,
        cr=config.cr,
        cs=config.cs
        )

    if config.weights:
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    if(self.best_val_metric == "rre" or self.best_val_metric == "rte"):
      self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.device = torch.device('cpu')
    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")
  
  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()
          self._save_checkpoint(
            epoch=epoch,
            val_dict=val_dict,
            best_val_metric=self.best_val_metric
          )

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if(self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
          if self.best_val < val_dict[self.best_val_metric]:
            logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )
        elif(self.best_val_metric == "rre" or self.best_val_metric == "rte" ):
          if self.best_val > val_dict[self.best_val_metric]:
            logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )

  def _save_checkpoint(self, epoch ,val_dict,best_val_metric,filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    content = f"{best_val_metric}_{val_dict[best_val_metric]}"
    filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

    logging.info(f"Saving checkpoint: {filename} ...")
    torch.save(state, filename)

class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    # if val_data_loader is not None:
    #   assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    trans = trans.to(pts.device)
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  
  # ---- current valid ----
  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    # self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter,success_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = next(data_loader_iter)
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput_src = input_dict['sinput_src'].to(self.device)
      # print(sinput_src.C)
      sinput_tgt = input_dict['sinput_tgt'].to(self.device)

      src_image = input_dict['src_range_image'].to(self.device)
      tgt_image = input_dict['tgt_range_image'].to(self.device)

      src_px, src_py= input_dict['src_px'], input_dict['src_py']
      tgt_px, tgt_py= input_dict['tgt_px'], input_dict['tgt_py']

      src_px = [s_x.to(self.device) for s_x in src_px]
      src_py = [s_y.to(self.device) for s_y in src_py]
      tgt_px = [t_x.to(self.device) for t_x in tgt_px]
      tgt_py = [t_y.to(self.device) for t_y in tgt_py]
              
      F0 = self.model(sinput_src,src_image,src_py,src_px)
      F1 = self.model(sinput_tgt,tgt_image,tgt_py,tgt_px)
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['sinput_src'].C, input_dict['sinput_tgt'].C, input_dict['tsfm']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0[:,:3], xyz1[:,:3], F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0[:,:3], xyz1[:,:3], weight=None)
      # print(loss)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[0, :3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[0, :3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
      if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success_meter.update(1)
      else:
        success_meter.update(0)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt[0,:,:], thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg,
        "success":success_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = torch.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
    # print('posF0',posF0.shape,'subF1',subF1.shape)
    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  # ---- current train ----
  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = next(data_loader_iter)
        data_time += data_timer.toc(average=False)

        sinput_src = input_dict['sinput_src'].to(self.device)
        sinput_tgt = input_dict['sinput_tgt'].to(self.device)

        src_image = input_dict['src_range_image'].to(self.device)
        # print('srcimage',src_image)
        tgt_image = input_dict['tgt_range_image'].to(self.device)
        # print('tgtimage',tgt_image)
        src_px, src_py= input_dict['src_px'], input_dict['src_py']
        tgt_px, tgt_py= input_dict['tgt_px'], input_dict['tgt_py']

        src_px = [s_x.to(self.device) for s_x in src_px]
        src_py = [s_y.to(self.device) for s_y in src_py]
        tgt_px = [t_x.to(self.device) for t_x in tgt_px]
        tgt_py = [t_y.to(self.device) for t_y in tgt_py]
                
        F0 = self.model(sinput_src,src_image,src_py,src_px)
        F1 = self.model(sinput_tgt,tgt_image,tgt_py,tgt_px)

        pos_pairs = input_dict['correspondence']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs[0,:,:],
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)
        
        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        # print(loss)
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()


