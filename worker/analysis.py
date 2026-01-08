# /workspace/competition_xai/worker/analysis.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from config.config import Config


class Analyzer:
    def __init__(self, config: Config, results: Dict[str, Any]):
        self.config = config
        self.results = results

        self.run_dir = Path(results["run_dir"])
        sub = f"test_{config.test_mode}" if config.mode == "test" else config.mode

        self.out_dir = self.run_dir / "analysis" / sub
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Entry points
    # =========================================================
    def train_generate_reports(self):
        self.save_json(self.results, self.out_dir / "train_results.json")

        history = self.results["history"]
        self.plot_history(history)

        print(f"[analysis.py] Training reports saved to: {self.out_dir}")

    def test_generate_reports(self):
        self.save_json(self.results, self.out_dir / "test_results.json")

        preds = self.results["preds"]
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        y_prob = preds["y_prob"]

        # (필수) Trainer.test()에서 아래 키를 같이 넘겨주셔야 grid를 그릴 수 있습니다.
        # preds["paths"] : List[str]
        # preds["datasets"] : List[str] (선택)
        paths = preds["paths"]
        datasets = preds.get("datasets", None)

        metrics = self.compute_metrics(y_true, y_pred)
        self.save_json(metrics, self.out_dir / "metrics.json")

        cm = confusion_matrix(y_true, y_pred)
        np.save(self.out_dir / "confusion_matrix.npy", cm)
        self.plot_confusion_matrix(cm, self.out_dir / "confusion_matrix.png")

        conf_stats = self.compute_confidence_stats(y_true, y_pred, y_prob)
        self.save_json(conf_stats, self.out_dir / "confidence_stats.json")

        roc_info = self.compute_roc_auc(y_true, y_prob)
        self.save_json(roc_info, self.out_dir / "roc_auc.json")
        if roc_info["task"] == "binary":
            self.plot_roc_curve_binary(
                y_true=y_true,
                y_prob=y_prob,
                out_path=self.out_dir / "roc_curve.png",
                auc_val=roc_info["roc_auc"],
            )

        self.plot_metric_bars(metrics, self.out_dir / "metrics_bar.png")
        self.plot_per_class_bars(metrics, self.out_dir / "per_class_bar.png")

        self.save_prediction_grid(
            paths=paths,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            datasets=datasets,
            out_path=self.out_dir / "examples_grid.png",
            n=16,
            cols=4,
        )

        self.save_prediction_grid(
            paths=paths,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            datasets=datasets,
            out_path=self.out_dir / "examples_wrong_grid.png",
            n=16,
            cols=4,
            only_wrong=True,
        )

        if datasets is not None:
            self.save_test_results_by_dataset(preds)

        print(f"[analysis.py] Test reports saved to: {self.out_dir}")

    def save_test_results_by_dataset(self, preds: Dict[str, Any]):
        datasets = preds.get("datasets", None)
        if datasets is None:
            return

        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        y_prob = preds["y_prob"]
        paths = preds["paths"]

        root = self.out_dir.parent / "test_datasets"
        root.mkdir(parents=True, exist_ok=True)

        # 등장 순서 유지하면서 유니크 dataset 목록 만들기
        seen = set()
        uniq = []
        for d in datasets:
            if d not in seen:
                seen.add(d)
                uniq.append(d)

        for ds in uniq:
            idxs = [i for i, d in enumerate(datasets) if d == ds]
            if len(idxs) == 0:
                continue

            out_dir = root / str(ds)
            out_dir.mkdir(parents=True, exist_ok=True)

            # subset preds 구성
            sub_preds = {
                "y_true": [y_true[i] for i in idxs],
                "y_pred": [y_pred[i] for i in idxs],
                "y_prob": [y_prob[i] for i in idxs],
                "paths": [paths[i] for i in idxs],
                "datasets": [datasets[i] for i in idxs],
            }

            # vis_matrix에서 읽기 쉽게 "run_dir"과 "test_mode"도 넣어줌
            sub_results = {
                "run_dir": str(self.run_dir),
                "mode": "test",
                "test_mode": self.config.test_mode,  # "id" or "ood"
                "dataset": str(ds),
                "preds": sub_preds,
            }

            self.save_json(sub_results, out_dir / "test_results.json")

    def evidence_harvesting_generate_reports(self):
        self.save_json(self.results, self.out_dir / "evidence_harvesting_results.json")

    def llm_debate_generate_reports(self):
        self.save_json(self.results, self.out_dir / "llm_debate_results.json")

    # =========================================================
    # Core plots / metrics
    # =========================================================
    def plot_history(self, history: List[Dict[str, Any]]):
        epochs = [h["epoch"] + 1 for h in history]
        train_loss = [h["train_loss"] for h in history]
        valid_loss = [h["valid_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        valid_acc = [h["valid_acc"] for h in history]

        fig = plt.figure()
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, valid_loss, label="valid_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        self.save_figure(fig, self.out_dir / "loss_curve.png")

        fig = plt.figure()
        plt.plot(epochs, train_acc, label="train_acc")
        plt.plot(epochs, valid_acc, label="valid_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        self.save_figure(fig, self.out_dir / "acc_curve.png")

    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))

        rep = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        out: Dict[str, Any] = {
            "count": int(len(y_true)),
            "accuracy": acc,
            "macro_precision": float(rep["macro avg"]["precision"]),
            "macro_recall": float(rep["macro avg"]["recall"]),
            "macro_f1": float(rep["macro avg"]["f1-score"]),
            "weighted_precision": float(rep["weighted avg"]["precision"]),
            "weighted_recall": float(rep["weighted avg"]["recall"]),
            "weighted_f1": float(rep["weighted avg"]["f1-score"]),
            "per_class": {},
        }

        for k, v in rep.items():
            if isinstance(k, str) and k.isdigit():
                out["per_class"][k] = {
                    "precision": float(v["precision"]),
                    "recall": float(v["recall"]),
                    "f1": float(v["f1-score"]),
                    "support": int(v["support"]),
                }

        return out

    def plot_confusion_matrix(self, cm: np.ndarray, out_path: Path):
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, values_format="d", colorbar=True)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        fig.tight_layout()
        self.save_figure(fig, out_path)

    def compute_confidence_stats(
        self, y_true: List[int], y_pred: List[int], y_prob: List[List[float]]
    ) -> Dict[str, Any]:
        probs = np.asarray(y_prob, dtype=np.float32)
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)

        conf = probs.max(axis=1)
        correct = y_true_np == y_pred_np

        wrong = ~correct
        return {
            "mean_conf_all": float(conf.mean()) if conf.size > 0 else None,
            "mean_conf_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_conf_wrong": float(conf[wrong].mean()) if wrong.any() else None,
        }

    def compute_roc_auc(
        self, y_true: List[int], y_prob: List[List[float]]
    ) -> Dict[str, Any]:
        y_true_np = np.asarray(y_true, dtype=np.int64)
        probs = np.asarray(y_prob, dtype=np.float32)
        n_classes = int(probs.shape[1])

        if n_classes == 2:
            auc_val = float(roc_auc_score(y_true_np, probs[:, 1]))
            return {"task": "binary", "roc_auc": auc_val, "pos_class": 1}
        else:
            auc_val = float(
                roc_auc_score(
                    y_true_np,
                    probs,
                    multi_class="ovr",
                    average="macro",
                )
            )
            return {
                "task": "multiclass",
                "roc_auc_ovr_macro": auc_val,
                "n_classes": n_classes,
            }

    def plot_roc_curve_binary(
        self,
        y_true: List[int],
        y_prob: List[List[float]],
        out_path: Path,
        auc_val: Optional[float] = None,
    ):
        y_true_np = np.asarray(y_true, dtype=np.int64)
        probs = np.asarray(y_prob, dtype=np.float32)
        scores = probs[:, 1]

        fpr, tpr, _ = roc_curve(y_true_np, scores)

        fig = plt.figure()
        label = "ROC" if auc_val is None else f"ROC (AUC={auc_val:.4f})"
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        self.save_figure(fig, out_path)

    def plot_metric_bars(self, metrics: Dict[str, Any], out_path: Path):
        names = [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]
        vals = [metrics[n] for n in names]

        fig = plt.figure(figsize=(10, 4))
        plt.bar(names, vals)
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("score")
        plt.tight_layout()
        self.save_figure(fig, out_path)

    def plot_per_class_bars(self, metrics: Dict[str, Any], out_path: Path):
        per = metrics["per_class"]  # {"0": {...}, "1": {...}, ...}
        classes = sorted(per.keys(), key=lambda x: int(x))

        precision = [per[c]["precision"] for c in classes]
        recall = [per[c]["recall"] for c in classes]
        f1 = [per[c]["f1"] for c in classes]

        x = np.arange(len(classes))
        w = 0.25

        fig = plt.figure(figsize=(10, 4))
        plt.bar(x - w, precision, width=w, label="precision")
        plt.bar(x, recall, width=w, label="recall")
        plt.bar(x + w, f1, width=w, label="f1")
        plt.xticks(x, classes)
        plt.ylim(0.0, 1.0)
        plt.xlabel("class")
        plt.ylabel("score")
        plt.legend()
        plt.tight_layout()
        self.save_figure(fig, out_path)

    # =========================================================
    # Image grid (test examples)
    # =========================================================
    def save_prediction_grid(
        self,
        paths: List[str],
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[List[float]],
        datasets: Optional[List[str]],
        out_path: Path,
        n: int = 16,
        cols: int = 4,
        only_wrong: bool = False,
    ):
        probs = np.asarray(y_prob, dtype=np.float32)
        conf = probs.max(axis=1)

        idxs = list(range(len(paths)))
        if only_wrong:
            idxs = [i for i in idxs if int(y_true[i]) != int(y_pred[i])]
        idxs = idxs[:n]

        if len(idxs) == 0:
            # 빈 플롯 파일이라도 남기고 싶으면 주석 해제
            fig = plt.figure(figsize=(4, 2))
            plt.axis("off")
            plt.title("No samples to display")
            self.save_figure(fig, out_path)
            return

        cols = max(1, min(cols, len(idxs)))

        rows = (len(idxs) + cols - 1) // cols  # ceil
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

        axes = np.array(axes).reshape(-1)

        for ax_i, ax in enumerate(axes):
            if ax_i >= len(idxs):
                ax.axis("off")
                continue

            i = idxs[ax_i]
            img = self.load_image(paths[i])

            ax.imshow(img)
            ax.axis("off")

            ds = ""
            if datasets is not None:
                ds = f"{datasets[i]} | "

            title = f"{ds}t={y_true[i]} p={y_pred[i]} conf={conf[i]:.3f}"
            ax.set_title(title, fontsize=9)

        fig.tight_layout()
        self.save_figure(fig, out_path)

    def load_image(self, path: str) -> np.ndarray:
        if self.config.input_modality == "rgb":
            img = Image.open(path).convert("RGB")
            return np.asarray(img)
        else:
            img = Image.open(path).convert("L")
            return np.asarray(img)

    # =========================================================
    # IO utils
    # =========================================================
    def save_json(self, obj: Dict[str, Any], path: Path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def save_figure(self, fig, path: Path):
        fig.savefig(path, dpi=200)
        plt.close(fig)
