import os
import random
from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

def mask_to_tensor(mask):
    """
    将 PIL mask 转为 [H, W] 的 long 张量
    """
    t = transforms.PILToTensor()(mask)  # [1, H, W], uint8
    return t.squeeze(0).long()

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_dir    = kwargs['data_dir']
        self.batch_size  = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.image_size  = kwargs['img_size']

        # 验证集：只做 Resize + ToTensor + Normalize
        self.val_image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            mask_to_tensor,
        ])

    def prepare_data(self):
        VOCSegmentation(self.data_dir, year="2012", image_set="train", download=False)
        VOCSegmentation(self.data_dir, year="2012", image_set="val", download=False)

    def setup(self, stage: Optional[str] = None):
        base_train = VOCSegmentation(
            self.data_dir, year="2012", image_set="train", download=False)

        base_val = VOCSegmentation(
            self.data_dir, year="2012", image_set="val", download=False)

        # 定义一个 joint transform：先 Resize，再随机水平/垂直翻转，最后张量化
        def train_joint_transform(img, mask):
            # 1) Resize
            img  = TF.resize(img,  (self.image_size, self.image_size), InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.image_size, self.image_size), InterpolationMode.NEAREST)
            # 2) Random flips
            if random.random() < 0.5:
                img  = TF.hflip(img);  mask = TF.hflip(mask)
            if random.random() < 0.5:
                img  = TF.vflip(img);  mask = TF.vflip(mask)
            # 3) ToTensor & Normalize / ToTensor mask
            img  = TF.to_tensor(img)
            img  = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            mask = mask_to_tensor(mask)
            return img, mask

        # Wrapper dataset，用于同时对 img 和 mask 做增强
        class VOCWrapper(torch.utils.data.Dataset):
            def __init__(self, base_ds, joint_transform, val_transform=None):
                self.base_ds = base_ds
                self.joint_transform = joint_transform
                self.val_transform = val_transform
            def __len__(self):
                return len(self.base_ds)
            def __getitem__(self, idx):
                img, mask = self.base_ds[idx]
                if self.val_transform is None:
                    return self.joint_transform(img, mask)
                else:
                    # 验证时只做 val_transform
                    img  = self.val_transform[0](img)  # list of two: img_transform, mask_transform
                    mask = self.val_transform[1](mask)
                    return img, mask

        # 组装 train/val datasets
        self.train_dataset = VOCWrapper(base_train, train_joint_transform)
        self.val_dataset   = VOCWrapper(base_val, None, val_transform=(self.val_image_transform, self.val_mask_transform))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()
