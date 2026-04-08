from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(name: str, use_pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model

    elif name == "res18res50":
        return Res18Res50Regressor(
            pretrained=use_pretrained,
            num_outputs=num_outputs,
        )

    else:
        raise ValueError(f"Unsupported backbone: {name}")


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        weight_path: str,
        use_pretrained: bool = False,
        feature_mode: str = "blocks",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.model = build_backbone(backbone_name, use_pretrained=use_pretrained)
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.feature_mode = feature_mode
        self.layer_names: list[str] = []
        self._features: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks: list = []
        self._register_hooks()

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._features[name] = out.detach().cpu()
        return hook

    def _register_hooks(self):
        # ---------- fusion model ----------
        if isinstance(self.model, Res18Res50Regressor):
            if self.feature_mode == "blocks":
                self.layer_names = [
                    "res18_layer1",
                    "res18_layer2",
                    "res18_layer3",
                    "res18_layer4",
                    "res18_pool",
                    "res50_layer1",
                    "res50_layer2",
                    "res50_layer3",
                    "res50_layer4",
                    "res50_pool",
                    "fusion_concat",
                ]
                return

            elif self.feature_mode == "all_conv":
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        safe_name = name.replace(".", "_").replace("-", "_")
                        self.layer_names.append(safe_name)
                        self._hooks.append(module.register_forward_hook(self._make_hook(safe_name)))

                # 這三個不是 conv module，但通常也很值得一起存
                self.layer_names.extend(["res18_pool", "res50_pool", "fusion_concat"])
                return

            else:
                raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        # ---------- single model ----------
        if self.feature_mode == "blocks":
            candidates = OrderedDict([
                ("layer1", self.model.layer1),
                ("layer2", self.model.layer2),
                ("layer3", self.model.layer3),
                ("layer4", self.model.layer4),
                ("avgpool", self.model.avgpool),
            ])
            for name, module in candidates.items():
                self.layer_names.append(name)
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

        elif self.feature_mode == "all_conv":
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    safe_name = name.replace(".", "_").replace("-", "_")
                    self.layer_names.append(safe_name)
                    self._hooks.append(module.register_forward_hook(self._make_hook(safe_name)))
            self.layer_names.append("avgpool")
            self._hooks.append(self.model.avgpool.register_forward_hook(self._make_hook("avgpool")))

        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor):
        # ---------- fusion + blocks ----------
        if isinstance(self.model, Res18Res50Regressor) and self.feature_mode == "blocks":
            feats = self.model.forward_features(x)
            return OrderedDict((k, v.detach().cpu()) for k, v in feats.items())

        # ---------- fusion + all_conv ----------
        if isinstance(self.model, Res18Res50Regressor) and self.feature_mode == "all_conv":
            self._features = OrderedDict()
            _ = self.model(x)  # hooks 會抓到所有 conv，model 也會更新 _last_features

            # 補上非 conv 的高層特徵
            extra_keys = ["res18_pool", "res50_pool", "fusion_concat"]
            for k in extra_keys:
                self._features[k] = self.model._last_features[k].detach().cpu()

            return self._features

        # ---------- single model ----------
        self._features = OrderedDict()
        _ = self.model(x)
        return self._features


class Res18Res50Regressor(nn.Module):
    def __init__(self, pretrained: bool = False, num_outputs: int = 1):
        super().__init__()

        weights18 = models.ResNet18_Weights.DEFAULT if pretrained else None
        weights50 = models.ResNet50_Weights.DEFAULT if pretrained else None

        res18 = models.resnet18(weights=weights18)
        res50 = models.resnet50(weights=weights50)

        # -------- ResNet18 backbone --------
        self.res18_stem = nn.Sequential(
            res18.conv1,
            res18.bn1,
            res18.relu,
            res18.maxpool,
        )
        self.res18_layer1 = res18.layer1
        self.res18_layer2 = res18.layer2
        self.res18_layer3 = res18.layer3
        self.res18_layer4 = res18.layer4
        self.res18_pool = res18.avgpool

        # -------- ResNet50 backbone --------
        self.res50_stem = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
        )
        self.res50_layer1 = res50.layer1
        self.res50_layer2 = res50.layer2
        self.res50_layer3 = res50.layer3
        self.res50_layer4 = res50.layer4
        self.res50_pool = res50.avgpool

        fusion_dim = 512 + 2048
        self.head = nn.Linear(fusion_dim, num_outputs)

        self._last_features: OrderedDict[str, torch.Tensor] = OrderedDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_dict = self.forward_features(x)
        fused = feat_dict["fusion_concat"]
        out = self.head(fused)
        return out

    def _flatten_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)

    def forward_features(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feats: OrderedDict[str, torch.Tensor] = OrderedDict()

        # ----- ResNet18 branch -----
        x18 = self.res18_stem(x)
        x18 = self.res18_layer1(x18)
        feats["res18_layer1"] = x18

        x18 = self.res18_layer2(x18)
        feats["res18_layer2"] = x18

        x18 = self.res18_layer3(x18)
        feats["res18_layer3"] = x18

        x18 = self.res18_layer4(x18)
        feats["res18_layer4"] = x18

        x18 = self.res18_pool(x18)
        x18 = self._flatten_if_needed(x18)
        feats["res18_pool"] = x18

        # ----- ResNet50 branch -----
        x50 = self.res50_stem(x)
        x50 = self.res50_layer1(x50)
        feats["res50_layer1"] = x50

        x50 = self.res50_layer2(x50)
        feats["res50_layer2"] = x50

        x50 = self.res50_layer3(x50)
        feats["res50_layer3"] = x50

        x50 = self.res50_layer4(x50)
        feats["res50_layer4"] = x50

        x50 = self.res50_pool(x50)
        x50 = self._flatten_if_needed(x50)
        feats["res50_pool"] = x50

        # ----- Fusion -----
        fusion = torch.cat([x18, x50], dim=1)
        feats["fusion_concat"] = fusion

        self._last_features = feats
        return feats