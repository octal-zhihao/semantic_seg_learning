import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(image, mask, pred=None):
    """
    输入：
        image: Tensor [C,H,W], unnormalized
        mask: Tensor [H,W], Ground Truth
        pred: Tensor [H,W], 模型预测（可选）
    """
    img = image.permute(1,2,0).cpu().numpy()
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
    fig, axes = plt.subplots(1, 3 if pred is not None else 2, figsize=(12,4))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(mask.cpu(), cmap="jet", vmin=0, vmax=20)
    axes[1].set_title("Ground Truth")
    if pred is not None:
        axes[2].imshow(pred.cpu(), cmap="jet", vmin=0, vmax=20)
        axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
