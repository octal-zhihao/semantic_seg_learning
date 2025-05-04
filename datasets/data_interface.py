import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Dataset
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
        self.val_split   = kwargs['val_split']
        self.augment     = kwargs['augment']

        # 只在 default 下用到
        self.val_img_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.val_msk_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            mask_to_tensor,
        ])

    def prepare_data(self):
        VOCSegmentation(self.data_dir, year="2012", image_set="train", download=False)
        VOCSegmentation(self.data_dir, year="2012", image_set="val", download=False)

    def setup(self, stage: Optional[str] = None):
        # 合并 trainval 然后再自己 split
        full = VOCSegmentation(self.data_dir, year="2012", image_set="trainval", download=False)
        n_val   = int(len(full) * self.val_split)
        n_train = len(full) - n_val
        train_base, val_base = random_split(full, [n_train, n_val])

        def joint_transform(img, mask):
            # 1. Resize
            img  = TF.resize(img,  (self.image_size, self.image_size), InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.image_size, self.image_size), InterpolationMode.NEAREST)

            # 2. Light 几何增强：随机翻转
            if self.augment in ("light", "strong"):
                if random.random() < 0.5:
                    img, mask = TF.hflip(img), TF.hflip(mask)
                if random.random() < 0.5:
                    img, mask = TF.vflip(img), TF.vflip(mask)

            # 3. Strong 额外增强：随机旋转 + 色彩抖动 + 随机裁剪
            if self.augment == "strong":
                # 随机旋转
                angle = random.uniform(-30, 30)
                img, mask = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR), \
                            TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                # 随机裁剪（再恢复至目标尺寸）
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    img, scale=(0.7, 1.0), ratio=(3/4, 4/3)
                )
                img = TF.resized_crop(img, i, j, h, w, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
                mask= TF.resized_crop(mask,i, j, h, w, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)
                # 色彩抖动
                color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                img = color_jitter(img)

            # 4. ToTensor + Normalize / ToTensor mask
            img  = TF.to_tensor(img)
            img  = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            mask = mask_to_tensor(mask)
            return img, mask

        # wrapper
        class VOCWrapper(Dataset):
            def __init__(self, base_ds, tf_fn):
                self.base_ds = base_ds
                self.tf_fn   = tf_fn
            def __len__(self):
                return len(self.base_ds)
            def __getitem__(self, idx):
                img, mask = self.base_ds[idx]
                # default 模式退化成只做验证 transformations
                if self.tf_fn is None:
                    img = self.val_img_tf(img)
                    mask = self.val_msk_tf(mask)
                    return img, mask
                return self.tf_fn(img, mask)

        train_tf = joint_transform if self.augment != "default" else None
        val_tf   = None  # always use default resize+tensor on val

        self.train_dataset = VOCWrapper(train_base, train_tf)
        self.val_dataset   = VOCWrapper(val_base,   val_tf)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()
