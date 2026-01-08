# /workspace/competition_xai/utils/llm_utils.py

import base64
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

import numpy as np
import torch
from PIL import Image
import open_clip


def image_to_data_url_png(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


def _extract_text_from_responses(resp_json: dict) -> str:
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
    max_retries: int = 6,
) -> str:
    api_key = (
        api_key or os.environ.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )
    assert api_key, "Missing OpenAI API key (env: openai_api_key or OPENAI_API_KEY)."

    contents = [{"type": "input_text", "text": prompt}]
    for p in image_paths:
        # 파일 존재/파일 여부를 강제 체크 (디렉토리 '.' 같은 실수를 즉시 잡음)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Invalid image path: {p}")
        contents.append({"type": "input_image", "image_url": image_to_data_url_png(p)})

    payload = {
        "model": model,
        "input": [{"role": "user", "content": contents}],
        "max_output_tokens": int(max_output_tokens),
    }

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(max_retries):
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if r.status_code == 200:
            return _extract_text_from_responses(r.json())

        # 429/503은 재시도 대상 (문서도 backoff 권장) :contentReference[oaicite:1]{index=1}
        if r.status_code in (429, 503):
            try:
                body = r.json()
            except Exception:
                body = {"raw_text": r.text}

            retry_after = r.headers.get("Retry-After")
            wait_s = None
            if retry_after is not None:
                try:
                    wait_s = float(retry_after)
                except Exception:
                    wait_s = None

            if wait_s is None:
                # 지수 백오프(1,2,4,8,16...) + 상한
                wait_s = float(min(60, 2**attempt))

            print(
                f"[openai_vision_explain] HTTP {r.status_code} attempt={attempt+1}/{max_retries}\n"
                f"Headers Retry-After={retry_after}\n"
                f"Body={body}\n"
                f"Sleeping {wait_s:.1f}s then retry..."
            )
            time.sleep(wait_s)
            last_err = (r.status_code, body)
            continue

        # 그 외는 즉시 실패(바디 출력)
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text}
        raise RuntimeError(f"OpenAI API error HTTP {r.status_code}: {body}")

    raise RuntimeError(f"OpenAI API error after retries: {last_err}")
