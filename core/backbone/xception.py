# /workspace/competition_xai/core/xception.py
import timm
import torch
import torch.nn as nn

from config.config import Config


def build_model(cfg: Config) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        in_chans=cfg.in_channels,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        global_pool=cfg.global_pool,
    )

    if cfg.ckpt_path is not None and cfg.ckpt_path != "" and cfg.mode != "train":
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model
