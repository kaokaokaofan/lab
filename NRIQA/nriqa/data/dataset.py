from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet的RGB統計量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_lines(txt_path: str | Path) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_scores(txt_path: str | Path) -> list[float]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [float(line.strip()) for line in f if line.strip()]


def to_uint8_image(img_path: str | Path) -> Image.Image:
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float32)

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    a_min = float(arr.min())
    a_max = float(arr.max())

    if a_max - a_min < 1e-12:
        arr_u8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = (arr - a_min) / (a_max - a_min)
        arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr_u8, mode="L")


def build_transform(use_pretrained_norm: bool = False):
    tfms = [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if use_pretrained_norm:
        tfms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(tfms)


class MRIQADataset(Dataset):
    def __init__(
        self,
        image_txt: str | Path,
        mos_txt: str | Path,
        image_root: str | Path,
        use_pretrained_norm: bool = False,
        transform=None,
    ):
        self.image_paths = read_lines(image_txt)
        self.scores = read_scores(mos_txt)

        if len(self.image_paths) != len(self.scores):
            raise ValueError(
                f"image數量 {len(self.image_paths)} 和 score數量 {len(self.scores)} 不一致"
            )

        self.image_root = Path(image_root)
        self.transform = transform or build_transform(use_pretrained_norm)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        img_path = self.image_root / rel_path

        image = to_uint8_image(img_path)
        image = self.transform(image)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return image, score, str(img_path)
