import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
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
        self.data_dir = kwargs['data_dir']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.image_size = kwargs['img_size']
        self.val_split = kwargs.get('val_split', 0.2)
        # 图像预处理：Resize + ToTensor + Normalize
        self.train_image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                              interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_image_transform = self.train_image_transform

        # mask 预处理：Resize (nearest) + 转为 long
        self.train_mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=InterpolationMode.NEAREST),
            mask_to_tensor,
        ])
        self.val_mask_transform = self.train_mask_transform

    def prepare_data(self):
        # 下载 VOC2012 数据集
        VOCSegmentation(self.data_dir, year="2012", image_set="train", download=False)

    def setup(self, stage: Optional[str] = None):
        # 整个训练集（train + val split）
        full = VOCSegmentation(
            self.data_dir,
            year="2012",
            image_set="train",
            transform=self.train_image_transform,
            target_transform=self.train_mask_transform,
        )
        # 划分训练/验证集
        n_val = int(len(full) * self.val_split)
        n_train = len(full) - n_val
        self.train_dataset, self.val_dataset = random_split(full, [n_train, n_val])

        # 测试集使用官方 val split
        self.test_dataset = VOCSegmentation(
            self.data_dir,
            year="2012",
            image_set="val",
            transform=self.val_image_transform,
            target_transform=self.val_mask_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )