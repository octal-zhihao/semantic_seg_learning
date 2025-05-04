#!/usr/bin/env python3
# analyze_voc_distribution.py

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from collections import defaultdict

def main(voc_root: str, year: str = "2012", splits=("train", "val")):
    # 初始化计数容器
    cls_image_count = defaultdict(int)
    cls_pixel_count = defaultdict(int)
    
    # 类别名称（VOC 共 21 类，0=background）
    VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "potted plant",
        "sheep", "sofa", "train", "tv/monitor"
    ]
    
    # 遍历 train 和 val 两个 split
    for split in splits:
        ds = VOCSegmentation(
            root=voc_root,
            year=year,
            image_set=split,
            download=False,
            transform=None,
            target_transform=None
        )
        print(f"Processing {len(ds)} images in {split} set...")
        
        for idx in range(len(ds)):
            # 只加载 mask
            _, mask = ds[idx]
            mask_np = np.array(mask, dtype=np.int32)  # shape (H, W)
            
            # 对每个类别统计
            present = set(np.unique(mask_np))
            # 忽略 void (255)
            if 255 in present:
                present.remove(255)
            
            for cls in present:
                if 0 <= cls < len(VOC_CLASSES):
                    cls_image_count[cls] += 1
            
            # 累加像素数
            for cls in range(len(VOC_CLASSES)):
                cls_pixel_count[cls] += int((mask_np == cls).sum())
    
    # 准备数据
    classes = VOC_CLASSES
    img_counts = [cls_image_count[i] for i in range(len(classes))]
    px_counts = [cls_pixel_count[i] for i in range(len(classes))]
    
    # 打印表格
    print(f"{'Class':<15} {'#Images':>10} {'#Pixels':>15}")
    for name, ic, pc in zip(classes, img_counts, px_counts):
        print(f"{name:<15} {ic:>10} {pc:>15}")
    
    # 可视化
    x = np.arange(len(classes))
    width = 0.35
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # 图片数柱状图
    axes[0].bar(x, img_counts, width)
    axes[0].set_title("Number of Images Containing Each Class")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha="right")
    axes[0].set_ylabel("Image Count")
    
    # 像素数柱状图（可选对数坐标）
    axes[1].bar(x, px_counts, width)
    axes[1].set_yscale("log")
    axes[1].set_title("Total Pixel Count for Each Class (log scale)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_ylabel("Pixel Count (log)")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--voc_root", type=str, default="./data", help="VOC2012 root directory")
    p.add_argument("--year", type=str, default="2012", help="VOC year")
    p.add_argument("--splits", nargs="+", default=["train"], help="Which splits to include")
    args = p.parse_args()
    main(args.voc_root, args.year, args.splits)
