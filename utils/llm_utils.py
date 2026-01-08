# /workspace/competition_xai/utils/llm_utils.py

import base64
import os
from pathlib import Path
from typing import List, Optional, Tuple

import requests


from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image

# pip install open_clip_torch
import open_clip


@torch.no_grad()
def clip_embed_image(
    image_path: Path,
    model_name: str = "ViT-B-32",
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


def image_to_data_url_png(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


def _extract_text_from_responses(resp_json: dict) -> str:
    # Responses API는 output 구조가 변형될 수 있어서 최대한 안전하게 텍스트만 모읍니다.
    texts: List[str] = []

    out = resp_json.get("output", [])
    for item in out:
        for c in item.get("content", []):
            t = c.get("type", "")
            if t in ("output_text", "text"):
                if "text" in c and isinstance(c["text"], str):
                    texts.append(c["text"])

    if texts:
        return "\n".join(texts).strip()

    # fallback
    if "text" in resp_json and isinstance(resp_json["text"], str):
        return resp_json["text"].strip()

    return ""


def openai_vision_explain(
    prompt: str,
    image_paths: List[Path],
    model: str,
    api_key: Optional[str] = None,
    max_output_tokens: int = 300,
    timeout: int = 120,
) -> str:
    """
    Responses API로 (텍스트 + 이미지들) 입력 → 설명 텍스트 반환.
    image_paths는 1~N개 가능(여기서는 원본+gradcam 2장 권장).
    """
    api_key = (
        api_key or os.environ.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )
    assert api_key, "Missing OpenAI API key (env: openai_api_key or OPENAI_API_KEY)."

    contents = [{"type": "input_text", "text": prompt}]
    for p in image_paths:
        contents.append({"type": "input_image", "image_url": image_to_data_url_png(p)})

    payload = {
        "model": model,
        "input": [{"role": "user", "content": contents}],
        "max_output_tokens": int(max_output_tokens),
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return _extract_text_from_responses(r.json())


def openai_embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[List[float], int]:
    """
    Embeddings API로 텍스트 임베딩 벡터(list[float])와 dim 반환.
    """
    api_key = (
        api_key or os.environ.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )
    assert api_key, "Missing OpenAI API key (env: openai_api_key or OPENAI_API_KEY)."

    payload = {"model": model, "input": text}
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    j = r.json()
    emb = j["data"][0]["embedding"]
    return emb, int(len(emb))
