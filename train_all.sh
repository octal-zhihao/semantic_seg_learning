#!/usr/bin/env bash

BACKBONES=(
  "deeplabv3_resnet50"
  "fcn_resnet50"
  "lraspp"
  "unet_resnet50"
  "unet_mobilenet_v2"
  "unet_efficientnet-b0"
  "Segformer"
  "DeepLabV3Plus"
  "UnetPlusPlus"
)

for B in "${BACKBONES[@]}"; do
  echo "=== Training with backbone: $B ==="
  python main.py \
    --backbone $B
done
