# model_module.py
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics
from config import NUM_CLASSES, LR

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.train_miou = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, ignore_index=255)
        self.val_miou = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, ignore_index=255)

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.squeeze(1).long())
        self.train_miou(logits.argmax(dim=1), y.squeeze(1))
        self.log("train_loss", loss)
        self.log("train_mIoU", self.train_miou, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.squeeze(1).long())
        self.val_miou(logits.argmax(dim=1), y.squeeze(1))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mIoU", self.val_miou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)
