# /workspace/competition_xai/utils/llm_utils.py

import base64
from pathlib import Path


def image_to_data_url_png(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")
