import torch
import pytorch_lightning as pl
from torchvision import models
import torch.nn.functional as F
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        num_classes = self.hparams.num_classes
        # select model backbone
        if self.hparams.backbone == "deeplabv3_resnet50":
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        elif self.hparams.backbone == "deeplabv3_resnet101":
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        elif self.hparams.backbone == "fcn_resnet50":
            self.model = models.segmentation.fcn_resnet50(pretrained=False, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        elif self.hparams.backbone == "fcn_resnet101":
            self.model = models.segmentation.fcn_resnet101(pretrained=False, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        elif self.hparams.backbone.startswith("unet_"):
            encoder = self.hparams.backbone.split("unet_")[-1]
            self.model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                classes=num_classes,
                activation=None
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.hparams.backbone}")

        # metrics
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")
        self.val_iou   = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")
        self.test_iou  = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")

    def forward(self, x):
        out = self.model(x)
        return out['out'] if isinstance(out, dict) else out

    def shared_step(self, batch, stage: str):
        imgs, masks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        preds = torch.argmax(logits, dim=1)
        if stage == "train":
            self.train_iou(preds, masks)
            self.log("train_mIoU", self.train_iou, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.val_iou(preds, masks)
            self.log("val_mIoU", self.val_iou, on_epoch=True, prog_bar=True)
        else:
            self.test_iou(preds, masks)
            self.log("test_mIoU", self.test_iou, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage=="val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def on_train_epoch_end(self):
        # log current learning rate
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [scheduler]