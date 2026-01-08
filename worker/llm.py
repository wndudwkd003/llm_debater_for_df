# /workspace/competition_xai/worker/llm.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.cm as cm

from config.config import Config
from core.builder import ModelBuilder
from core.gradcam import GradCAM, find_last_conv2d
from utils.data_utils import get_debate_dataset, get_evidence_harvesting_dataset
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def write_json(path: Path, obj: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overlay_cam_on_rgb(
    rgb_uint8: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    heat = (cm.jet(cam_01)[..., :3] * 255).astype(np.uint8)
    out = rgb_uint8.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def tensor_chw_to_rgb_uint8(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().cpu().float().clamp(0, 1)
    x = (x * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()
    return x


def select_topk(
    records: List[Dict[str, Any]], correct: bool, k: int
) -> List[Dict[str, Any]]:
    # correct / wrong 먼저 필터
    filt = [r for r in records if bool(r["correct"]) == bool(correct)]
    # confidence 내림차순
    filt.sort(key=lambda r: float(r["confidence"]), reverse=True)

    # 라벨별로 분리 (binary 가정: 0/1)
    by_label: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for r in filt:
        y = int(r["y_true"])
        if y in by_label:
            by_label[y].append(r)

    # 0/1 반반 목표
    k0 = k // 2
    k1 = k - k0  # 홀수면 1쪽에 1개 더

    take0 = by_label[0][:k0]
    take1 = by_label[1][:k1]
    out = take0 + take1

    # 부족하면 남은 것들에서 confidence 순으로 채우기
    if len(out) < k:
        used_ids = {id(r) for r in out}
        rest = [r for r in filt if id(r) not in used_ids]
        out += rest[: (k - len(out))]

    return out[:k]


class LLMDebater:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        run_dir = Path(self.config.ckpt_path)
        if run_dir.suffix == ".pt":
            run_dir = run_dir.parent
        self.run_dir = run_dir

        self.best_ckpt_path = self.run_dir / "best.pt"
        self.config.ckpt_path = str(self.best_ckpt_path)

        self.model: Optional[nn.Module] = None
        self.cam: Optional[GradCAM] = None

        # run_dir/evidence/<input_modality>/
        self.evidence_root = self.run_dir / "evidence" / self.config.input_modality
        self.evidence_root.mkdir(parents=True, exist_ok=True)

    def build_model(self) -> nn.Module:
        model = ModelBuilder.build(self.config)
        model.to(self.device)
        model.eval()
        return model

    def build_cam(self, model: nn.Module) -> GradCAM:
        target_layer = find_last_conv2d(model)
        return GradCAM(model=model, target_layers=[target_layer])

    @torch.no_grad()
    def run_inference_collect(self, loader: DataLoader, split: str):
        records = []

        for inputs, targets, meta in tqdm(loader, desc=f"[EVIDENCE:{split}]"):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values

            y_true = targets.detach().cpu().tolist()
            y_pred = pred.detach().cpu().tolist()
            y_prob = probs.detach().cpu().tolist()
            confidence = conf.detach().cpu().tolist()

            paths = meta["path"]
            datasets = meta.get("dataset", ["unknown"] * len(paths))
            indices = meta["index"]

            bs = len(y_true)
            for i in range(bs):
                yt = int(y_true[i])
                yp = int(y_pred[i])
                cf = float(confidence[i])

                records.append(
                    {
                        "split": split,
                        "dataset": str(datasets[i]),
                        "index": int(indices[i]),
                        "path": str(paths[i]),
                        "y_true": yt,
                        "y_pred": yp,
                        "y_prob": y_prob[i],
                        "confidence": cf,
                        "correct": bool(yt == yp),
                    }
                )

        return records

    def save_images_with_gradcam(
        self,
        loader: DataLoader,
        out_dir: Path,
        selected_records: List[Dict[str, Any]],
    ):
        img_root = out_dir / "images"
        corr_dir = img_root / "correct"
        wrong_dir = img_root / "wrong"
        corr_dir.mkdir(parents=True, exist_ok=True)
        wrong_dir.mkdir(parents=True, exist_ok=True)

        by_path = {r["path"]: r for r in selected_records}

        for inputs, _, meta in tqdm(loader, desc="[SAVE:GRADCAM]"):
            inputs = inputs.to(self.device, non_blocking=True)

            paths = meta["path"]
            sel = [i for i, p in enumerate(paths) if str(p) in by_path]
            if not sel:
                continue

            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            targets = [
                ClassifierOutputTarget(int(c)) for c in pred.detach().cpu().tolist()
            ]

            with torch.enable_grad():
                cams = self.cam(inputs, targets=targets)  # numpy (B,H,W)

            for i in sel:
                p = str(paths[i])
                r = by_path[p]

                ds = r["dataset"]
                idx = r["index"]
                yt = r["y_true"]
                yp = r["y_pred"]
                cf = float(r["confidence"])
                ok = bool(r["correct"])

                stem = f"{ds}__{idx}__y{yt}__p{yp}__c{cf:.4f}"
                save_dir = corr_dir if ok else wrong_dir

                out_img = save_dir / f"{stem}.png"
                out_cam = save_dir / f"{stem}_gradcam.png"

                rgb = tensor_chw_to_rgb_uint8(inputs[i])
                Image.fromarray(rgb).save(out_img)

                cam_01 = np.clip(cams[i].astype(np.float32), 0.0, 1.0)
                over = overlay_cam_on_rgb(rgb, cam_01, alpha=0.45)
                Image.fromarray(over).save(out_cam)

                r["saved_image"] = str(out_img)
                r["saved_gradcam"] = str(out_cam)

    def harvest_evidence(self) -> Dict[str, Any]:
        # 설정
        split = "train"
        topk = self.config.evidence_top_k

        # 모델/캠 준비
        self.model = self.build_model()
        self.cam = self.build_cam(self.model)

        # harvest용 로더(ID train / OOD train)
        id_loader, ood_loader = get_evidence_harvesting_dataset(self.config)

        summary: Dict[str, Any] = {
            "mode": "evidence_harvesting",
            "run_dir": str(self.run_dir),
            "ckpt_path": str(self.best_ckpt_path),
            "evidence_root": str(self.evidence_root),
            "input_modality": self.config.input_modality,
            "split": split,
            "topk": int(topk),
            "id": {},
            "ood": {},
        }

        for mode, loader in [("id", id_loader), ("ood", ood_loader)]:
            records = self.run_inference_collect(loader, split=split)

            # dataset별로 분리 저장
            by_dataset = {}
            for r in records:
                by_dataset.setdefault(r["dataset"], []).append(r)

            mode_root = self.evidence_root / mode
            mode_root.mkdir(parents=True, exist_ok=True)

            mode_entry: Dict[str, Any] = {}
            for dataset_name, ds_records in by_dataset.items():
                out_dir = mode_root / dataset_name / split
                out_dir.mkdir(parents=True, exist_ok=True)

                manifest_path = out_dir / "manifest.jsonl"
                write_jsonl(manifest_path, ds_records)

                topk_correct = select_topk(ds_records, correct=True, k=topk)
                topk_wrong = select_topk(ds_records, correct=False, k=topk)
                selected = topk_correct + topk_wrong

                write_jsonl(out_dir / "topk_correct.jsonl", topk_correct)
                write_jsonl(out_dir / "topk_wrong.jsonl", topk_wrong)

                # 선택 샘플에 대해서만 원본+gradcam 저장
                self.save_images_with_gradcam(loader, out_dir, selected)

                write_json(
                    out_dir / "summary.json",
                    {
                        "mode": mode,
                        "dataset": dataset_name,
                        "split": split,
                        "count": int(len(ds_records)),
                        "correct_count": int(
                            sum(1 for r in ds_records if r["correct"])
                        ),
                        "wrong_count": int(
                            sum(1 for r in ds_records if not r["correct"])
                        ),
                        "topk": int(topk),
                        "files": {
                            "manifest": str(manifest_path),
                            "topk_correct": str(out_dir / "topk_correct.jsonl"),
                            "topk_wrong": str(out_dir / "topk_wrong.jsonl"),
                            "images_dir": str(out_dir / "images"),
                        },
                    },
                )

                mode_entry[dataset_name] = {
                    "out_dir": str(out_dir),
                    "count": int(len(ds_records)),
                }

            summary[mode] = mode_entry

        write_json(self.evidence_root / "summary.json", summary)
        return summary

    def call_llm_on_saved_evidence(self) -> Dict[str, Any]:
        raise NotImplementedError("LLM calling pipeline is not implemented yet.")

    def llm_debate(self) -> Dict[str, Any]:
        # test용 debate loader는 여기서 사용 예정
        _ = get_debate_dataset(self.config)
        raise NotImplementedError("llm_debate is not implemented yet.")
