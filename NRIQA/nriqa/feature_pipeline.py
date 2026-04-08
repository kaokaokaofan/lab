from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nriqa.models.backbone import FeatureExtractor, get_device


@torch.no_grad()
def extract_dataset_features(feature_extractor: FeatureExtractor, loader: DataLoader):
    collected = {name: [] for name in feature_extractor.layer_names}
    collected["scores"] = []
    collected["paths"] = []

    for images, scores, paths in loader:
        images = images.to(feature_extractor.device, non_blocking=True)
        feats = feature_extractor.forward_features(images)

        for key in feature_extractor.layer_names:
            collected[key].append(feats[key].numpy())
        collected["scores"].append(scores.numpy())
        collected["paths"].extend(paths)

    for key in feature_extractor.layer_names:
        collected[key] = np.concatenate(collected[key], axis=0)
    collected["scores"] = np.concatenate(collected["scores"], axis=0)
    return collected


def save_feature_dict(feature_dict, layer_names: list[str], save_dir: str | Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key in layer_names:
        np.save(save_dir / f"{key}.npy", feature_dict[key])
    np.save(save_dir / "scores.npy", feature_dict["scores"])

    with open(save_dir / "paths.txt", "w", encoding="utf-8") as f:
        for p in feature_dict["paths"]:
            f.write(str(p) + "\n")

    with open(save_dir / "layer_names.json", "w", encoding="utf-8") as f:
        json.dump(layer_names, f, indent=2, ensure_ascii=False)
