# semantic_seg_learning

## Project Structure

```bash
root/
├── data/                       # 存放原始 VOC2012 数据集 (按 VOC 格式)
├── datasets/
│   └── data_interface.py       # LightningDataModule 实现
├── model/
│   └── model_interface.py      # LightningModule 实现
├── utils.py                    # 常用工具函数，比如可视化
└── main.py                     # 训练脚本入口
```