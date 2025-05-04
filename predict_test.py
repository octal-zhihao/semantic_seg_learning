# predict_test.py

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from model.model_interface import MInterface  # 确保 import 路径正确

class VOCTestDataset(Dataset):
    def __init__(self, voc_root: str, image_size: int):
        """
        voc_root: VOC2012 根目录，包含 JPEGImages 和 ImageSets/Main
        """
        self.img_dir = os.path.join(voc_root, "JPEGImages")
        ids_txt = os.path.join(voc_root, "ImageSets", "Segmentation", "test.txt")
        with open(ids_txt, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return img_id, img
def get_pascal_palette():
    """
    生成 PASCAL VOC 256-color 调色板 (list of length 768)，
    前 21*3 个值对应类别 0–20 的 RGB 颜色，后面自动补零。
    """
    palette = []
    for label in range(256):
        r = g = b = 0
        cid = label
        for i in range(8):
            r |= ((cid >> 0) & 1) << (7 - i)
            g |= ((cid >> 1) & 1) << (7 - i)
            b |= ((cid >> 2) & 1) << (7 - i)
            cid >>= 3
        palette.extend([r, g, b])
    return palette


def main():
    parser = argparse.ArgumentParser(description="Predict VOC2012 test set")
    parser.add_argument("--voc_root",  type=str, default="data/VOCdevkit/VOC2012",
                        help="Path to VOCdevkit/VOC2012 root")
    parser.add_argument("--checkpoint", type=str, 
                        default="mycheckpoints/unet_efficientnet-b0_epoch=76_val_mIoU=0.56.ckpt",
                        help="Path to best .ckpt file")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Resize test images to this size")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="comp5_test_cls",
                        help="Where to save predicted masks")
    args = parser.parse_args()

    # 1. Load model from checkpoint
    model = MInterface.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()

    # 2. Prepare test dataset & dataloader
    test_ds = VOCTestDataset(args.voc_root, args.image_size)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # 3. Create output dir
    os.makedirs(args.out_dir, exist_ok=True)
    # 4. Predict & save masks
    # 生成调色板
    palette = get_pascal_palette()
    # 只取前 21 类的
    palette_21 = palette[:21*3]
    with torch.no_grad():
        for batch in loader:
            img_ids, imgs = batch
            imgs = imgs.to(model.device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # [B, H, W]

            for img_id, mask in zip(img_ids, preds):
                # mask is a 2D array of class indices (uint8)
                im = Image.fromarray(mask.astype(np.uint8), mode="P")
                im.putpalette(palette_21)
                out_path = os.path.join(args.out_dir, img_id + ".png")
                im.save(out_path)

    print(f"Saved predictions for {len(test_ds)} images to '{args.out_dir}/'")

if __name__ == "__main__":
    main()
