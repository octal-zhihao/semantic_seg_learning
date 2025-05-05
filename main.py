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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--img_size", type=int, default=224, help="Resize image to this size")
    parser.add_argument("--augment", type=str, default="default", choices=["light", "strong", "default"], help="Data augmentation type")
    # parser.add_argument("--use_mixup", action="store_true", help="Enable mixup augmentation")
    # parser.add_argument("--use_cutmix", action="store_true", help="Enable cutmix augmentation")

    # 模型参数
    parser.add_argument(
        "--backbone",
        type=str,
        default="unet_efficientnet-b0",
        choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", "lraspp",
                 "unet_resnet50", "unet_resnet101", "unet_mobilenet_v2", "unet_efficientnet-b0", "unet_efficientnet-b3",
                 "Segformer", "DeepLabV3Plus", "UnetPlusPlus"],
        help="Model backbone/type",
    )
    parser.add_argument("--num_classes", type=int, default=21, help="Number of classes")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # 调度器参数
    parser.add_argument("--T_max", type=int, default=100, help="Max number of epochs for cosine annealing")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    # 训练参数
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of total epochs")
    parser.add_argument("--precision", type=int, default=32, help="Floating point precision (16 or 32)")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Trainer accelerator: cpu/gpu/mps")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    # 日志与回调
    parser.add_argument("--project", type=str, default="semantic-seg", help="WandB project name")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--save_top_k", type=int, default=3, help="Number of top checkpoints to save")

    return parser.parse_args()

def train():
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
        filename=f"{args['backbone']}" + "_{epoch:02d}_{val_mIoU:.2f}"
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
    # print("训练完成，自动评估测试集...")
    # model_path = checkpoint_cb.best_model_path
    # if model_path:
    #     best_model = MInterface.load_from_checkpoint(model_path)
    #     trainer.test(best_model, datamodule=data_module)
    # else:
    #     print("未保存最佳模型，使用当前模型测试...")
    #     trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    train()
