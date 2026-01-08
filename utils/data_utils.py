# /workspace/competition_xai/utils/data_utils.py

import json
from pathlib import Path
from typing import Dict, List

import datasets
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from config.config import Config


def read_jsonl(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_image_path(dataset_dir: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return dataset_dir / pp


def _load_image_as_tensor(
    img_path: Path, img_size: int, input_modality: str
) -> torch.Tensor:
    if input_modality == "rgb":
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.open(img_path).convert("L")

    if img_size is not None and img_size > 0:
        img = img.resize((img_size, img_size), resample=Image.Resampling.BILINEAR)

    arr = np.array(img)

    if input_modality == "rgb":
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        x = torch.from_numpy(arr).unsqueeze(0).float() / 255.0

    return x


class JsonlImageDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        jsonl_path: Path,
        img_size: int,
        input_modality: str,
        dataset_name: str,
        debug_mode: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.jsonl_path = jsonl_path
        self.img_size = img_size
        self.input_modality = input_modality
        self.dataset_name = dataset_name
        self.rows = read_jsonl(jsonl_path)
        self.debug_mode = debug_mode

    def __len__(self) -> int:
        if self.debug_mode:
            return min(200, len(self.rows))
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_path = _resolve_image_path(self.dataset_dir, r["path"])
        y = int(r["label"])
        x = _load_image_as_tensor(img_path, self.img_size, self.input_modality)

        meta = {
            "path": str(img_path),
            "dataset": self.dataset_name,
            "index": int(idx),
        }
        return x, y, meta


def _list_dataset_names(datasets_root: Path) -> List[str]:
    names = [p.name for p in datasets_root.iterdir() if p.is_dir()]
    names.sort()
    return names


def _make_dataset(
    datasets_root: Path, name: str, split: str, config: Config
) -> JsonlImageDataset:
    ddir = datasets_root / name
    jsonl_path = ddir / f"{split}.jsonl"
    return JsonlImageDataset(
        dataset_dir=ddir,
        jsonl_path=jsonl_path,
        img_size=config.img_size,
        input_modality=config.input_modality,
        dataset_name=name,
        debug_mode=config.DEBUG_MODE,
    )


def get_evidence_harvesting_dataset(config: Config):
    datasets_root = Path(config.datasets_path)

    # 먼저 ID
    id_ds = _make_dataset(datasets_root, config.train_dataset, "train", config)

    # 그 다음 OOD
    all_names = _list_dataset_names(datasets_root)
    ood_names = [n for n in all_names if n != config.train_dataset]
    ood_parts = [_make_dataset(datasets_root, n, "train", config) for n in ood_names]
    ood_ds = ConcatDataset(ood_parts)

    return DataLoader(
        id_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    ), DataLoader(
        ood_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_debate_dataset(config: Config):
    # 이거는 최종 토론을 통한 테스트 단계에서 사용하는 데이터 세트임
    # 이걸로 최종 평가하는 것

    datasets_root = Path(config.datasets_path)

    # all dataset test splits
    all_names = _list_dataset_names(datasets_root)
    parts = [_make_dataset(datasets_root, n, "test", config) for n in all_names]
    debate_ds = ConcatDataset(parts)

    return DataLoader(
        debate_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def get_train_loader(config: Config) -> DataLoader:
    datasets_root = Path(config.datasets_path)
    train_ds = _make_dataset(datasets_root, config.train_dataset, "train", config)

    return DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_valid_loader(config: Config) -> DataLoader:
    datasets_root = Path(config.datasets_path)
    valid_ds = _make_dataset(datasets_root, config.train_dataset, "valid", config)

    return DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_test_loader(config: Config):
    datasets_root = Path(config.datasets_path)

    if config.test_mode == "id":
        test_ds = _make_dataset(datasets_root, config.train_dataset, "test", config)

    elif config.test_mode == "ood":
        all_names = _list_dataset_names(datasets_root)
        ood_names = [n for n in all_names if n != config.train_dataset]
        parts = [_make_dataset(datasets_root, n, "test", config) for n in ood_names]
        test_ds = ConcatDataset(parts)

    else:
        raise ValueError(f"Unknown test_mode: {config.test_mode}")

    return DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
