# /workspace/competition_xai/utils/clip_utils.py

# pip install open_clip_torch
import open_clip
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def clip_embed_image(
    image_path: Path,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cpu",
) -> Tuple[List[float], int]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    feat = model.encode_image(x)  # (1, D)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.squeeze(0).detach().cpu().float().numpy().astype(np.float32)

    return vec.tolist(), int(vec.shape[0])
