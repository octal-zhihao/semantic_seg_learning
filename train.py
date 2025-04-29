# train.py
import torch
from model_module import SegmentationModel
from data_module import VOCDataModule
import pytorch_lightning as pl
from config import MAX_EPOCHS
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
def main():
    model = SegmentationModel()
    data_module = VOCDataModule()
    wandb_logger = WandbLogger(
        project="voc_segmentation",      # 项目名，自行修改
        name="fcn_resnet50_run",         # 实验名，可选
        log_model=True                   # 自动保存模型到 WandB
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mIoU',
        dirpath='mycheckpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)
    wandb.finish()

if __name__ == '__main__':
    main()
