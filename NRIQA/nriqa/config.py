from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    train_image_txt: str
    train_mos_txt: str
    train_root: str
    test_image_txt: str
    test_mos_txt: str
    test_root: str


@dataclass
class ModelConfig:
    backbone: Literal["resnet18", "resnet50", "res18res50"] = "resnet50"
    use_pretrained: bool = True
    feature_mode: Literal["blocks", "all_conv"] = "blocks"
    num_outputs: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    momentum: float = 0.9
    epochs: int = 5
    optimizer: Literal["sgd", "adam"] = "sgd"
    criterion: Literal["mse"] = "mse"
    pin_memory: bool = True
    save_dir: str = "checkpoints"
    save_name: str = "baseline"


@dataclass
class FeatureConfig:
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    feature_root: str = "features"


@dataclass
class SVRConfig:
    k_value: int = 100000
    n_components: int = 75
    layer_svr_c: float = 25.0
    layer_svr_gamma: str = "scale"
    layer_svr_epsilon: float = 0.1
    high_svr_c: float = 25.0
    high_svr_gamma: str = "scale"
    high_svr_epsilon: float = 0.1
    preselect_ascending: bool = True
    top_k_layers: int | None = None
    rank_by: Literal["plcc", "srcc"] = "plcc"


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================
# 統一在這裡改
# =========================
DATA = DataConfig(
    train_image_txt=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\train_images.txt",
    train_mos_txt=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\mos_train.txt",
    train_root=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\databases\train",
    test_image_txt=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\test_images.txt",
    test_mos_txt=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\mos_test.txt",
    test_root=r"C:\Users\USER\Desktop\fan\NRIQA\decoded_data\databases\test",
)

MODEL = ModelConfig(
    backbone="res18res50",
    use_pretrained=True,
    feature_mode="all_conv",
)

TRAIN = TrainConfig(
    batch_size=32,
    epochs=30,
    lr=1e-3,
    momentum=0.9,
    optimizer="sgd",
    save_dir="checkpoints",
    save_name=f"{MODEL.backbone}_baseline",
)

FEATURE = FeatureConfig(
    batch_size=16,
    feature_root="features",
)

SVR_CFG = SVRConfig(
    k_value=100000,
    n_components=75,
    layer_svr_c=25,
    layer_svr_gamma="scale",
    layer_svr_epsilon=0.1,
    high_svr_c=25,
    high_svr_gamma="scale",
    high_svr_epsilon=0.1,
    preselect_ascending=True,
    top_k_layers=3,      # 現在正式改成 top-3
    rank_by="plcc",      # 用 validation 的 PLCC 排
)


# =========================
# 自動產生路徑
# =========================
CHECKPOINT_PATH = rf"C:\Users\USER\Desktop\fan\NRIQA\checkpoints\{MODEL.backbone}_baseline_best.pth"
FEATURE_DIR = rf"{FEATURE.feature_root}\{MODEL.backbone}_{MODEL.feature_mode}"