import logging
import lib.transforms as t
import torch
from dataset.kitti import KITTIDataset


def make_data_loader(
        config,                     #
        phase,
        batch_size,
        num_threads=0,
        shuffle=None
):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  # the dataset name

  data_augmentation = False
  transforms = []
  if phase in ['train', 'trainval']:
    data_augmentation = True
    transforms += [t.Jitter()]

  dset = KITTIDataset(
      config,phase,data_augmentation=data_augmentation)

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=dset.collate_fn,
      pin_memory=True,
      drop_last=True)

  return loader
