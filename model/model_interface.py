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
        if self.hparams.backbone == "deeplabv3_resnet50":
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        elif self.hparams.backbone == "deeplabv3_resnet101":
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        elif self.hparams.backbone == "fcn_resnet50":
            self.model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        elif self.hparams.backbone == "fcn_resnet101":
            self.model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
            self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        elif self.hparams.backbone == "unet_resnet50":
            self.model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", classes=num_classes, activation=None)
        elif self.hparams.backbone == "unet_resnet101":
            self.model = smp.Unet(encoder_name="resnet101", encoder_weights="imagenet", classes=num_classes, activation=None)
        elif self.hparams.backbone == "unet_mobilenetv2":
            self.model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", classes=num_classes, activation=None)
        elif self.hparams.backbone == "unet_efficientnetb0":
            self.model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", classes=num_classes, activation=None)
        elif self.hparams.backbone == "unet_efficientnetb3":
            self.model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet", classes=num_classes, activation=None)
        else:
            raise ValueError(f"Unsupported backbone: {self.hparams.backbone}")
        # 监控 IoU
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")
        self.test_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro")

    def forward(self, x):
        out = self.model(x)
        # smp Unet returns masks, torchvision returns dict
        if isinstance(out, dict):
            return out['out']
        return out

    def shared_step(self, batch, stage: str):
        imgs, masks = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks, ignore_index=255)  # ignore_index=255 for void class
        preds = torch.argmax(logits, dim=1)
        if stage == "train":
            self.train_iou(preds, masks)
            self.log("train_mIoU", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.val_iou(preds, masks)
            self.log("val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.test_iou(preds, masks)
            self.log("test_mIoU", self.test_iou, on_step=False, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage=="val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
