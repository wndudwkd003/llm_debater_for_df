# /workspace/competition_xai/core/xception.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    timm_name: str = "xception"

    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 2

    # optional timm args
    drop_rate: float = 0.0
    global_pool: str = "avg"

    ckpt_path: Optional[Path] = None


def build_model(cfg: ModelConfig) -> nn.Module:
    model = timm.create_model(
        cfg.timm_name,
        pretrained=cfg.pretrained,
        in_chans=cfg.in_channels,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        global_pool=cfg.global_pool,
    )

    if cfg.ckpt_path is not None:
        ckpt = torch.load(str(cfg.ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt, strict=True)

    return model
