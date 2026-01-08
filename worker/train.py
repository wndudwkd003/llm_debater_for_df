# /workspace/competition_xai/worker/train.py

from pathlib import Path
from config.config import Config
from utils.date_utils import get_current_timestamp
from core.builder import ModelBuilder
from utils.data_utils import get_train_loader, get_valid_loader, get_test_loader
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from torch.nn import functional as F
import json


class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.device = torch.device(config.device)

        run_dir = Path(self.config.run_dir)
        ts = get_current_timestamp()
        mn = f"{self.config.model_name}_{self.config.input_modality}"
        run_dir /= f"{ts}_{mn}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = run_dir
        print(f"[train.py] Run directory created at: {run_dir}")
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.no_improve_epochs = 0

        self.es_patience = config.early_stopping_patience
        self.es_delta = config.early_stopping_delta

        self.best_ckpt_path = self.run_dir / "best.pt"

        self.model: nn.Module = None
        self.train_loader: DataLoader = None
        self.valid_loader: DataLoader = None
        self.test_loader: DataLoader = None

        self.optimizer: AdamW = None
        self.scheduler: CosineAnnealingLR = None

        ####

    def update_early_stopping(self, valid_loss: float, epoch: int) -> bool:
        """
        Returns:
            stop (bool): True면 학습 중단
        """
        improved = (self.best_loss - valid_loss) > self.es_delta

        if improved:
            self.best_loss = valid_loss
            self.best_epoch = epoch
            self.no_improve_epochs = 0

            # best 저장
            self.save_checkpoint(self.best_ckpt_path, epoch, valid_loss)
            print(
                f"[train.py] Saved best checkpoint: {self.best_ckpt_path} (valid_loss={valid_loss:.6f})"
            )
            return False

        # no improvement
        self.no_improve_epochs += 1
        if self.no_improve_epochs >= self.es_patience:
            print(
                f"[train.py] Early stopping triggered. "
                f"best_epoch={self.best_epoch+1}, best_valid_loss={self.best_loss:.6f}"
            )
            return True

        return False

    def save_checkpoint(self, path: Path, epoch: int, valid_loss: float):
        torch.save(self.model.state_dict(), str(path))

        meta = {
            "epoch": epoch,
            "valid_loss": float(valid_loss),
            "config": vars(self.config),
        }

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)

    def _load_model(self):
        model = ModelBuilder.build(self.config)
        return model

    def run_epoch(
        self,
        loader: DataLoader,
        split: str,
    ):
        is_train = split == "train"

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(loader, desc=f"[{split.upper()}]")

        if is_train:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            for batch in pbar:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # stats
                bs = targets.size(0)
                total_loss += loss.item() * bs
                total_count += bs

                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()

                avg_loss = total_loss / total_count
                avg_acc = total_correct / total_count
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        avg_loss = total_loss / max(total_count, 1)
        avg_acc = total_correct / max(total_count, 1)
        return float(avg_loss), float(avg_acc)

    def train(self):
        model = self._load_model()
        model.to(self.device)
        self.model = model

        self.train_loader = get_train_loader(self.config)
        self.valid_loader = get_valid_loader(self.config)
        self.test_loader = get_test_loader(self.config)

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
        )

        history = []

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.run_epoch(self.train_loader, "train")
            valid_loss, valid_acc = self.run_epoch(self.valid_loader, "valid")

            self.scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                }
            )

            print(
                f"[epoch {epoch+1}/{self.config.num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

            # early stopping + best.pt 저장
            stop = self.update_early_stopping(valid_loss, epoch)
            if stop:
                break

        print(
            f"[train.py] Training finished. best_epoch={self.best_epoch+1}, best_valid_loss={self.best_loss:.6f}"
        )

        return {
            "mode": "train",
            "run_dir": str(self.run_dir),
            "best_ckpt_path": str(self.best_ckpt_path),
            "best_epoch": int(self.best_epoch),
            "best_valid_loss": float(self.best_loss),
            "history": history,
        }

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        y_true = []
        y_pred = []
        y_prob = []

        for batch in tqdm(loader, desc="[TEST]"):
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == targets).sum().item()
            total_count += bs

            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

        avg_loss = total_loss / max(total_count, 1)
        avg_acc = total_correct / max(total_count, 1)

        return {
            "loss": float(avg_loss),
            "acc": float(avg_acc),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "count": int(total_count),
        }

    def test(self):
        self.model = self._load_model()
        self.model.to(self.device)

        self.test_loader = get_test_loader(self.config)

        test_eval = self.evaluate_loader(self.test_loader)

        return {
            "mode": "test",
            "run_dir": str(self.run_dir),
            "test_mode": self.config.test_mode,
            "ckpt_path": self.config.ckpt_path,
            "metrics": {
                "loss": test_eval["loss"],
                "acc": test_eval["acc"],
                "count": test_eval["count"],
            },
            "preds": {
                "y_true": test_eval["y_true"],
                "y_pred": test_eval["y_pred"],
                "y_prob": test_eval["y_prob"],
            },
        }
