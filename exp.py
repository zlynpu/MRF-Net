import numpy as np
import matplotlib.pyplot as plt
from lib.benchmark_utils import to_o3d_pcd

def do_range_projection(points_xyz, points_refl):
        H,W = (64,2048)

        # points_xyz,points_refl = (points_xyz[self.point_valid_index != -1],
        #                           points_refl[self.point_valid_index != -1]) \
        #              if self.point_valid_index is not None else (points_xyz,points_refl)

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

        px = proj_x.copy()
        py = proj_y.copy()

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)

        proj_range = np.zeros((H,W)) + 1e-5
        proj_cumsum = np.zeros((H,W)) + 1e-5
        proj_reflectivity = np.zeros((H, W))
        proj_range[proj_y,proj_x] += depth
        proj_cumsum[proj_y,proj_x] += 1
        # print(points_refl.shape)
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

        return refl_image, px, py

if __name__ == "__main__":
        block = np.fromfile('/home/huile/zhangliyuan/Dataset/Kitti/dataset/sequences/00/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)
        points_xyz = block[:,:3]
        pcd0 = to_o3d_pcd(points_xyz)
        pcd0 = pcd0.voxel_down_sample(0.3)
        print(np.array(pcd0.points).shape)
        pcd0 = pcd0.voxel_down_sample(0.3)
        print(np.array(pcd0.points).shape)
        points_refl = block[:,3]
        # print(points_refl.shape)
        range_image, px, py = do_range_projection(points_xyz, points_refl)
        # print(range_image.shape)
        plt.imsave('refl_image.png', range_image, cmap='inferno')