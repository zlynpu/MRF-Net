
import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=2)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=True)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.3)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='PointMinkUNet')
cs = [32,64,128,256,128,64,64,64]
# cs = [32,64,128,256,256,128,128,64,32]
net_arg.add_argument('--cs', type=list, default=cs, help='Feature dimension')
net_arg.add_argument('--cr', type=int, default=1)
# net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='rte',help='[feat_match_ratio,rre,rte,success]')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=200)
opt_arg.add_argument('--lr', type=float, default=0.1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)


misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument('--nn_max_n',type=int,default=500,help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
# ----------------------------------------------------------------------- #
# Kitti  ---- |output path|
output_kitti = "outputs_kitti/exp_pointminkunet_new"
logging_arg.add_argument('--out_dir', type=str, default=output_kitti)

#Kitti  ----  |resume dir|
misc_arg.add_argument('--resume_dir', type=str, default=None)

# kitti ---- |num thread|
misc_arg.add_argument('--train_num_thread', type=int, default=8)
misc_arg.add_argument('--val_num_thread', type=int, default=8)
misc_arg.add_argument('--test_num_thread', type=int, default=8)

# Kitti ---- |voxel size|
voxel_size_Kitti = 0.3
data_arg.add_argument('--voxel_size', type=float, default=voxel_size_Kitti)
overlap_radius = 0.45
data_arg.add_argument('--overlap_radius', type=float, default=overlap_radius)
range_size = [64,2048]
data_arg.add_argument('--range_size', type=list, default=range_size)
max_voxels = 84000
data_arg.add_argument('--max_voxels', type=int, default=max_voxels)
augment_noise = 0.01
data_arg.add_argument('--augment_noise',type=float,default=augment_noise)
augment_shift_range = 2.0
data_arg.add_argument('--augment_shift_range',type=float,default=augment_shift_range)
augment_scale_max = 1.2
data_arg.add_argument('--augment_scale_max',type=float,default=augment_scale_max)
augment_scale_min = 0.8
data_arg.add_argument('--augment_scale_min',type=float,default=augment_scale_min)
max_points = 512
data_arg.add_argument('--max_points',type=int,default=max_points)


kitti_path = "/home/huile/zhangliyuan/Dataset/Kitti"
data_arg.add_argument('--root', type=str, default=kitti_path)




def get_config():
  args = parser.parse_args()
  return args
