# train.py
import torch
from model_module import SegmentationModel
from data_module import VOCDataModule
import pytorch_lightning as pl
from config import MAX_EPOCHS

def main():
    model = SegmentationModel()
    data_module = VOCDataModule()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
