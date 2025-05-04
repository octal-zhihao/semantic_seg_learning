import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import DInterface
from model import MInterface
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--img_size", type=int, default=224, help="Resize image to this size")
    parser.add_argument("--augment", type=str, default="light", choices=["light", "strong", "default"], help="Data augmentation type")
    # parser.add_argument("--use_mixup", action="store_true", help="Enable mixup augmentation")
    # parser.add_argument("--use_cutmix", action="store_true", help="Enable cutmix augmentation")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="deeplabv3_resnet50", help="Model backbone")
    parser.add_argument("--num_classes", type=int, default=21, help="Number of classes")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # 训练参数
    parser.add_argument("--max_epochs", type=int, default=300, help="Number of total epochs")
    parser.add_argument("--precision", type=int, default=32, help="Floating point precision (16 or 32)")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Trainer accelerator: cpu/gpu/mps")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")

    # 日志与回调
    parser.add_argument("--project", type=str, default="semantic-seg", help="WandB project name")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--save_top_k", type=int, default=3, help="Number of top checkpoints to save")

    return parser.parse_args()

def main():
    args = vars(parse_args())
    # 1. Data
    data_module = DInterface(**args)
    # 2. Model
    model = MInterface(**args)

    # 3. Logger & Callbacks
    wandb_logger = WandbLogger(project="voc_semantic_segmentation")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_mIoU",
        dirpath='mycheckpoints/',
        mode="max",
        save_top_k=args["save_top_k"],
        filename="deeplabv3-{epoch:02d}-{val_iou:.3f}"
    )
    earlystop_cb = EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=args["early_stop_patience"],
        verbose=True
    )

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        precision=args["precision"],
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices="auto",
    )

    # 5. Train & Test
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()
