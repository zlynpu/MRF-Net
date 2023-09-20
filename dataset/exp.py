import torch,sys,os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/home/huile/zhangliyuan/Code/MRFNet'))

from dataset.kitti import KITTIDataset
from lib.utils import load_config
from easydict import EasyDict as edict

if __name__ == '__main__':
    config = load_config('/home/huile/zhangliyuan/Code/MRFNet/configs/train/kitti.yaml')
    config = edict(config)
    data = KITTIDataset(config=config, split='train')
    data_loader = DataLoader(dataset=data,
                            batch_size=config.batch_size, 
                            shuffle=True,
                            num_workers=config.num_workers,
                            collate_fn=data.collate_fn,
                            pin_memory=False,
                            drop_last=False)
    for batch,data in enumerate(data_loader):
        image = data['tgt_range_image']
        depth_image = image[0,1,:,:]
        plt.imsave('refl_image_data'+'.png', depth_image, cmap='inferno')
        print('image:',image.shape)
        px = data['tgt_px']
        # print('px:',px.shape)
        # pcd = data['pcd_src']
        sinput = data['sinput_src']
        corres = data['correspondence']
        rot = data['rot']
