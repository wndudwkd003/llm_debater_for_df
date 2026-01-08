# /workspace/competition_xai/vis_matrix.py

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score


DO_LIST = [
    {
        "name": "xception_rgb_KoDF",
        "path": "/workspace/competition_xai/runs/20260108_114358_xception_rgb_KoDF",
    },
    {
        "name": "xception_rgb_RVF",
        "path": "/workspace/competition_xai/runs/20260108_115026_xception_rgb_RVF",
    },
    {
        "name": "xception_rgb_TIMIT",
        "path": "/workspace/competition_xai/runs/20260108_115555_xception_rgb_TIMIT",
    },
]

OUT_DIR = "/workspace/competition_xai/exp"
METRICS_TO_PLOT = ["accuracy", "macro_f1", "roc_auc"]


# ----------------------------
# IO
# ----------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_train_dataset_from_run_dir(run_dir: Path) -> str:
    # 예: 20260108_115555_xception_rgb_TIMIT -> TIMIT
    parts = run_dir.name.split("_")
    return parts[-1] if parts else run_dir.name


def _list_available_test_datasets(run_dir: Path) -> List[str]:
    root = run_dir / "analysis" / "test_datasets"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "test_results.json").exists():
            out.append(p.name)
    return out


def _load_preds_from_dataset_json(
    run_dir: Path, dataset: str
) -> Optional[Dict[str, Any]]:
    """
    run_dir/analysis/test_datasets/<dataset>/test_results.json
    """
    p = run_dir / "analysis" / "test_datasets" / dataset / "test_results.json"
    if not p.exists():
        return None
    obj = _read_json(p)
    preds = obj.get("preds", None)
    if preds is None:
        return None
    return obj  # 전체 obj를 반환(표시용 meta: test_mode 등)


# ----------------------------
# Metric computation
# ----------------------------
def _compute_metric_from_obj(obj: Dict[str, Any], metric: str) -> Optional[float]:
    preds = obj["preds"]
    y_true = np.asarray(preds["y_true"], dtype=np.int64)
    y_pred = np.asarray(preds["y_pred"], dtype=np.int64)

    if y_true.size == 0:
        return None

    if metric == "accuracy":
        return float((y_true == y_pred).mean())

    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if metric == "roc_auc":
        y_prob = np.asarray(preds["y_prob"], dtype=np.float32)
        if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
            return None
        if y_prob.shape[1] == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))

    raise ValueError(f"Unknown metric: {metric}")


# ----------------------------
# Plot / Save
# ----------------------------
def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    cell_notes: Optional[List[List[str]]] = None,  # test_mode 같은 주석용
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = np.array(matrix, dtype=np.float32)
    mask = np.isnan(m)
    mm = np.ma.array(m, mask=mask)

    fig, ax = plt.subplots(
        figsize=(1.2 * len(col_labels) + 3, 1.0 * len(row_labels) + 3)
    )
    im = ax.imshow(mm, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            if mask[i, j]:
                txt = "-"
            else:
                txt = f"{m[i, j]:.3f}"
                if cell_notes is not None and cell_notes[i][j]:
                    txt = txt + "\n" + cell_notes[i][j]  # 예: "id"/"ood"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_csv(
    matrix: np.ndarray, row_labels: List[str], col_labels: List[str], out_path: Path
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(col_labels) + "\n")
        for r, row_name in enumerate(row_labels):
            vals = []
            for c in range(len(col_labels)):
                v = matrix[r, c]
                vals.append("" if np.isnan(v) else f"{v:.6f}")
            f.write(row_name + "," + ",".join(vals) + "\n")


# ----------------------------
# Main
# ----------------------------
def vis_matrix_all():
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(d["path"]) for d in DO_LIST]
    row_labels = [d["name"] for d in DO_LIST]

    # columns = (가능한 dataset들의 합집합) 우선 run명 기반 + 실제 폴더 기반을 합침
    col_set = []
    for rd in run_dirs:
        # run 디렉토리명 기반(예: KoDF/RVF/TIMIT)
        ds = _infer_train_dataset_from_run_dir(rd)
        if ds not in col_set:
            col_set.append(ds)
        # 실제 저장된 test_datasets 폴더들도 포함
        for ds2 in _list_available_test_datasets(rd):
            if ds2 not in col_set:
                col_set.append(ds2)

    col_labels = col_set

    for metric in METRICS_TO_PLOT:
        mat = np.full((len(run_dirs), len(col_labels)), np.nan, dtype=np.float32)
        notes: List[List[str]] = [["" for _ in col_labels] for __ in run_dirs]

        for r_i, rd in enumerate(run_dirs):
            for c_i, test_ds in enumerate(col_labels):
                obj = _load_preds_from_dataset_json(rd, test_ds)
                if obj is None:
                    continue

                val = _compute_metric_from_obj(obj, metric)
                mat[r_i, c_i] = np.nan if val is None else float(val)

                # 주석: id/ood 표시 (파일에 들어있는 값 그대로)
                tm = str(obj.get("test_mode", ""))
                if tm in ("id", "ood"):
                    notes[r_i][c_i] = tm

        png_path = out_root / f"matrix_{metric}.png"
        csv_path = out_root / f"matrix_{metric}.csv"

        _plot_heatmap(
            mat,
            row_labels=row_labels,
            col_labels=col_labels,
            title=f"Cross-Dataset Matrix ({metric})",
            out_path=png_path,
            cell_notes=notes,  # 셀마다 id/ood 같이 표시
        )
        _save_csv(mat, row_labels=row_labels, col_labels=col_labels, out_path=csv_path)

        print(f"[vis_matrix.py] saved: {png_path}")
        print(f"[vis_matrix.py] saved: {csv_path}")


if __name__ == "__main__":
    vis_matrix_all()
