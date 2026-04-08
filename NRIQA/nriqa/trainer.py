from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nriqa.models.backbone import build_backbone, get_device
from nriqa.utils.metrics import plcc, rmse, srcc


class RegressorTrainer:
    def __init__(self, model: nn.Module, optimizer, criterion, device=None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0.0

        for images, scores, _ in loader:
            images = images.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, scores)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []

        for images, scores, _ in loader:
            images = images.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)

            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, scores)
            total_loss += loss.item() * images.size(0)
            y_true.extend(scores.cpu().numpy().tolist())
            y_pred.extend(outputs.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        return {
            "loss": avg_loss,
            "rmse": rmse(y_true, y_pred),
            "plcc": plcc(y_true, y_pred),
            "srcc": srcc(y_true, y_pred),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }


def create_optimizer(model: nn.Module, name: str, lr: float, momentum: float = 0.9):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def create_criterion(name: str):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported criterion: {name}")


def save_checkpoint(model: nn.Module, save_dir: str | Path, save_name: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{save_name}.pth"
    torch.save(model.state_dict(), path)
    return path


