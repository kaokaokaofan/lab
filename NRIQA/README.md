# NRIQA Refactor

## 建議目錄

```text
nriqa_refactor/
├─ nriqa/
│  ├─ config.py
│  ├─ trainer.py
│  ├─ feature_pipeline.py
│  ├─ data/
│  │  └─ dataset.py
│  ├─ models/
│  │  └─ backbone.py
│  ├─ quality/
│  │  └─ svr_fusion.py
│  └─ utils/
│     └─ metrics.py
├─ scripts/
│  ├─ train.py
│  ├─ extract_features.py
│  └─ run_svr.py
└─ README.md
```

## 核心想法

- `dataset.py`：只管資料讀取與前處理
- `backbone.py`：統一建立 resnet18 / resnet50
- `trainer.py`：統一 train / eval
- `feature_pipeline.py`：統一特徵抽取與存檔
- `svr_fusion.py`：統一 PCA + layer SVR + high-level SVR
- `scripts/*.py`：只放「這次要跑什麼」

## 你之後怎麼切模型

在 `scripts/train.py` 改：

```python
MODEL = ModelConfig(backbone="resnet18", use_pretrained=True)
```

或

```python
MODEL = ModelConfig(backbone="resnet50", use_pretrained=True)
```

不用再複製整份程式。

## 之後如果要加 fusion

建議再新增：

- `nriqa/models/fusion.py`
- `scripts/train_fusion.py`
- `scripts/extract_fusion_features.py`

這樣單模型和 fusion 會共用同一套 dataset / metrics / svr pipeline。
