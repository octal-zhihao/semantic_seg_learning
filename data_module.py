# data_module.py
import os
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import *

class VOCDataset(VOCSegmentation):
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        super().__init__(root=root, year='2012', image_set=image_set, download=False)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class VOCDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = VOC_ROOT

        self.common_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])

    def setup(self, stage=None):
        self.train_dataset = VOCDataset(self.data_dir, image_set='train',
                                        transform=self.common_transform,
                                        target_transform=self.mask_transform)
        self.val_dataset = VOCDataset(self.data_dir, image_set='val',
                                      transform=self.common_transform,
                                      target_transform=self.mask_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
