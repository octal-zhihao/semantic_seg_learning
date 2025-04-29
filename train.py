# train.py
import torch
from model_module import SegmentationModel
from data_module import VOCDataModule
import pytorch_lightning as pl
from config import MAX_EPOCHS
from pytorch_lightning.loggers import WandBLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
def main():
    model = SegmentationModel()
    data_module = VOCDataModule()
    wandb.init(project='segmentation_model', entity='octal-zhihao')
    wandb_logger = WandBLogger(project='segmentation_model', log_model=True)
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
