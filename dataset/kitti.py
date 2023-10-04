# Basic libs
import os, glob, random, copy, sys
import numpy as np
import open3d
import torch
import logging
sys.path.append(os.path.abspath("/home/huile/zhangliyuan/Code/MRFNet"))
from scipy.spatial.transform import Rotation

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
import torchsparse.nn.functional as F

# Dataset parent class
from scipy.linalg import expm, norm
from dataset.collate import sparse_collate_fn
from torch.utils.data import Dataset
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_correspondences, to_tensor

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T

class KITTIDataset(Dataset):
    """
    We augment data with rotation, scaling, and translation
    Then we get correspondence, and voxelise them, 
    """
    DATA_FILES = {
        'train': '/home/huile/zhangliyuan/Code/MRFNet/configs/kitti/train_kitti.txt',
        'val': '/home/huile/zhangliyuan/Code/MRFNet/configs/kitti/val_kitti.txt',
        'test': '/home/huile/zhangliyuan/Code/MRFNet/configs/kitti/test_kitti.txt'
    }
    def __init__(self,config,split,data_augmentation=True,manual_seed=False):
        super(KITTIDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root,'dataset')
        self.icp_path = os.path.join(config.root,'icp')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.overlap_radius
        self.range_size = tuple(config.range_size)
        self.max_voxels = config.max_voxels
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min
        self.max_corr = config.max_points
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(split)
        self.split = split
        self.data = {}


    def prepare_kitti_ply(self, split):
        assert split in ['train','val','test']

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split in ['test','val']:
            self.files.remove((8, 15, 58))
        print(f'Num_{split}: {len(self.files)}')



    def __len__(self):
        return len(self.files)


    # def reset_seed(self, seed=0):
    #     logging.info(f"Resetting the data loader seed to {seed}")
    #     self.randg.seed(seed)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # extract xyz
        block0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyz0 = block0[:,:3]
        block1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz1 = block1[:,:3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                            @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                        open3d.registration.TransformationEstimationPointToPoint(),
                                                        open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        # get voxel and range image indices
        src_pcd_input = copy.deepcopy(xyz0)
        tgt_pcd_input = copy.deepcopy(xyz1)
        src_pcd_refl = block0[:,3]
        tgt_pcd_refl = block1[:,3]

        # self.point_valid_index = None
        # sel0 = self.do_voxel_projection(block0,src_pcd_input,'sinput_src')
        # self.data['sel0'] = sel0
        # self.do_range_projection(src_pcd_input,src_pcd_refl,sel0,'src_range_image','src_px','src_py')

        # self.point_valid_index = None
        # sel1 = self.do_voxel_projection(block1,tgt_pcd_input,'sinput_tgt')
        # self.data['sel1'] = sel1
        # self.do_range_projection(tgt_pcd_input,tgt_pcd_refl,sel1,'tgt_range_image','tgt_px','tgt_py')

        # add data augmentation
        matching_search_voxel_size = self.search_voxel_size
        scale = 1
        if(self.data_augmentation):
            # add gaussian noise
            # src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
            # tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

            # rotate the point cloud
            # euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            # euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            # rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            # if(np.random.rand(1)[0]>0.5):
            #     src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
            #     rot=np.matmul(rot,rot_ab.T)
            # else:
            #     tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
            #     rot=np.matmul(rot_ab,rot)
            #     trans=np.matmul(rot_ab,trans)
            
            # # scale the pcd
            # scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            # src_pcd_input = src_pcd_input * scale
            # tgt_pcd_input = tgt_pcd_input * scale
            # trans = scale * trans
            T0 = sample_random_trans(src_pcd_input, self.randg, 45)
            T1 = sample_random_trans(tgt_pcd_input, self.randg, 45)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            src_pcd_input = self.apply_transform(src_pcd_input, T0)
            tgt_pcd_input = self.apply_transform(tgt_pcd_input, T1)
            # matching_search_voxel_size = self.search_voxel_size
            if random.random() < 0.95:
                scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
                matching_search_voxel_size *= scale
                trans[:,3] *= scale
                src_pcd_input = scale * src_pcd_input
                tgt_pcd_input = scale * tgt_pcd_input

        else:
            trans = M2
        
        block0[:,:3] = src_pcd_input
        block1[:,:3] = tgt_pcd_input

        src_xyz_norm = src_pcd_input - src_pcd_input.min(0,keepdims=1)
        tgt_xyz_norm = tgt_pcd_input - tgt_pcd_input.min(0,keepdims=1)

        self.point_valid_index = None
        sel0 = self.do_voxel_projection(block0,src_xyz_norm,'sinput_src')
        self.data['sel0'] = sel0
        self.data['raw_pcd_src'] = src_pcd_input[sel0]
        self.do_range_projection(xyz0,src_pcd_refl,sel0,'src_range_image','src_px','src_py')

        self.point_valid_index = None
        sel1 = self.do_voxel_projection(block1,tgt_xyz_norm,'sinput_tgt')
        self.data['sel1'] = sel1
        self.data['raw_pcd_tgt'] = tgt_pcd_input[sel1]
        self.do_range_projection(xyz1,tgt_pcd_refl,sel1,'tgt_range_image','tgt_px','tgt_py')
        
        # get correspondence
                
        matching_inds = get_correspondences(to_o3d_pcd(src_pcd_input[sel0]), to_o3d_pcd(tgt_pcd_input[sel1]), trans, matching_search_voxel_size)
        if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
            return self.__getitem__(np.random.choice(len(self.files),1)[0])
        # print('matching:',matching_inds.shape)
        # rot, trans = to_tensor(rot), to_tensor(trans)
        trans = to_tensor(trans)
        
        self.data['correspondence'] = matching_inds
        self.data['tsfm'] = trans
        self.data['scale'] = scale
        return self.data

    def do_voxel_projection(self,feat,points_xyz,name):

        feat_one = torch.ones((feat.shape[0],1))
        pc = np.round(points_xyz / self.voxel_size).astype(np.int32)

        _, inds, inverse_map = sparse_quantize(pc, return_index=True,
                                              return_inverse=True)
        # todo: remove some voxels during training, 
        # so it is necessary to remove the point cloud corresponding to the voxel in this process
        # uses torchsparse to speed things up
        if self.split == 'train':
            # print(inds.shape)
            if len(inds) > self.max_voxels:
                inds = np.random.choice(inds,self.max_voxels,replace=False)
                pc_ = pc[inds]
                all_point = torch.concat([torch.from_numpy(pc),torch.zeros(pc.shape[0]).reshape(-1,1)],dim=1).int()
                voxel_valid = torch.concat([torch.from_numpy(pc_),torch.zeros(pc_.shape[0]).reshape(-1,1)],dim=1).int()

                old_hash = F.sphash(all_point) # 120000+
                sparse_hash = F.sphash(voxel_valid) # max 84000

                self.point_valid_index = F.sphashquery(old_hash,sparse_hash)
                # print(self.point_valid_index.shape)


        # todo: The number of fixed points helps turn px,py into a regular tensor, speeding up r2p,p2r
        # if self.point_valid_index.shape[0] > self.num_points :
        #   pass

        # coord_,feat_ = (points_xyz[self.point_valid_index != -1],
        #                        feat[self.point_valid_index != -1]) \
        #                 if self.point_valid_index is not None else (points_xyz,feat)
        coord_,feat_ = (points_xyz[inds],feat[inds])
        # print('coord_:',coord_.shape)
        self.data[name] = SparseTensor(feats=feat_,coords=coord_)
        return inds
        
            
    def do_range_projection(self, points_xyz, points_refl,sel, name_img, name_px, name_py):
        H,W = self.range_size if self.range_size is not None else (64,2048)

        # points_xyz,points_refl = (points_xyz[self.point_valid_index != -1],
        #                           points_refl[self.point_valid_index != -1]) \
        #              if self.point_valid_index is not None else (points_xyz,points_refl)

        points_xyz, points_refl = (points_xyz[sel],points_refl[sel])
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

        # px = proj_x.copy()
        # py = proj_y.copy()
        # Correct Out-of-Range Indices
        proj_y = np.clip(proj_y, 0, H - 1)  
        proj_x = np.clip(proj_x, 0, W - 1)  

        # Correct Out-of-Range Indices
        px = proj_x.copy()
        py = proj_y.copy()

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)
        # print(proj_y)

     

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

        self.data[name_img] = range_image
        self.data[name_px] = px
        self.data[name_py] = py
        
    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
    
    @staticmethod
    def collate_fn(inputs):
        '''
            self.data['sinput_src'] = SparseTensor(feat,coord)
            self.data['sinput_tgt'] = SparseTensor(feat,coord)
            self.data['correspondences'] = matching_inds # numpy
            self.data['tsfm'] = tsfm # tensor
            self.data['scale'] = scale # scale
            self.data['src_range_image']
            self.data['src_px']
            self.data['src_py']
            self.data['tgt_range_image']
            self.data['tgt_px']
            self.data['tgt_py']
        '''
        return sparse_collate_fn(inputs)
    


