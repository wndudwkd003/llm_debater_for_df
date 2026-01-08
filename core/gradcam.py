import torch.nn as nn

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def find_last_conv2d(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last


def run_gradcam(cam, inputs, class_ids):
    """
    inputs: torch.Tensor [B,C,H,W]
    class_ids: torch.Tensor [B] (각 샘플의 타겟 클래스 id)
    return: cams np.ndarray [B,H,W] in [0,1]
    """

    targets = [ClassifierOutputTarget(int(c.item())) for c in class_ids]
    grayscale_cam = cam(input_tensor=inputs, targets=targets)  # [B,H,W], float
    return grayscale_cam
