from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch

import sys
sys.path.append('/home/wanghuiwen/code/qarv-release-main')

from lvae.paths import known_datasets

__all__ = ['HsiDataset', 'get_hsi_dateset']
# 导入高光谱数据

class HsiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root # will be accessed by the training script
        self.transform = transform
        # scan and add images
        self.image_paths = sorted(Path(root).rglob('*.*'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = np.load(impath) #.astype(np.float32)
        img = img[10:40,:,:]
        img = np.transpose(img, (1,2,0))
        im = self.transform(img)
        return im

class OHSDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(self.root.rglob('*.npy'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'
        
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = np.load(impath)
        # OHS has 32 bands, slice first 30 to match pre-trained base codec
        img = img[0:30, :, :]
        img = np.transpose(img, (1, 2, 0)) # H, W, C
        im = self.transform(img)
        
        # Get label from parent directory name
        label = self.class_to_idx[impath.parent.name]
        return im, label



class PaviaDataset(Dataset):
    def __init__(self, root, transform=None, split='train', patch_size=64):
        self.root = Path(root)
        self.transform = transform
        self.patch_size = patch_size
        self.split = split
        
        # Load data
        img = np.load(self.root / 'Pavia_uint8.npy') # (1096, 715, 102)
        gt = np.load(self.root / 'Pavia_gt.npy')     # (1096, 715)
        
        # Slice to 30 bands to match base model
        img = img[:, :, :30]
        
        # Split: use top 80% for training, bottom 20% for testing
        H, W, C = img.shape
        split_h = int(H * 0.8)
        
        if split == 'train':
            self.img = img[:split_h, :, :]
            self.gt = gt[:split_h, :]
        else:
            self.img = img[split_h:, :, :]
            self.gt = gt[split_h:, :]
            
    def __len__(self):
        # We can simulate a large number of patches by sampling
        if self.split == 'train':
            return 8000
        else:
            return 1000

    def __getitem__(self, index):
        H, W, C = self.img.shape
        
        # Random crop
        y = np.random.randint(0, H - self.patch_size + 1)
        x = np.random.randint(0, W - self.patch_size + 1)
        
        patch_img = self.img[y:y+self.patch_size, x:x+self.patch_size, :]
        patch_gt = self.gt[y:y+self.patch_size, x:x+self.patch_size]
        
        if self.transform:
            patch_img = self.transform(patch_img)
            
        patch_gt = torch.from_numpy(patch_gt).long()
        return patch_img, patch_gt


def get_hsi_dateset(name: str, transform_cfg: str=None) -> Dataset:
    """ get image dataset from name

    Args:
        name (str): dataset name, see functions above
        transform_cfg (str, optional): config, example: 'crop=256,hflip=True'
    """
    # make input transform
    transform = []
    if transform_cfg is not None:
        transform_cfg = eval(f'dict({transform_cfg})')
        assert isinstance(transform_cfg, dict)
        if 'crop' in transform_cfg:
            t = tv.transforms.RandomCrop(transform_cfg['crop'], pad_if_needed=True, padding_mode='reflect')
            transform.append(t)
        if transform_cfg.get('hflip', False):
            t = tv.transforms.RandomHorizontalFlip(p=0.5)
            transform.append(t)
    transform.append(tv.transforms.ToTensor())
    transform = tv.transforms.Compose(transform)

    # find dataset root, and initialize dataset
    if name.startswith('ohs'):
        dataset = OHSDataset(root=known_datasets.get(name, name), transform=transform)
    elif name.startswith('pavia'):
        split = 'train' if 'train' in name else 'test'
        dataset = PaviaDataset(root=known_datasets.get(name, name), transform=transform, split=split)
    else:
        dataset = HsiDataset(root=known_datasets.get(name, name), transform=transform)
    return dataset

if __name__ == "__main__":
    trainset = get_hsi_dateset('hsi-train')
    dataloader = DataLoader(trainset, batch_size=8, drop_last=True, num_workers=8)
    data_iter = iter(dataloader)
    img = next(data_iter)
    print(img)
    print(img.shape,type(img))