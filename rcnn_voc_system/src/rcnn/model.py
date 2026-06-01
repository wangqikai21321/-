from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


class RCNNAlexNet(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        alexnet = models.alexnet(weights=weights)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.fc6 = nn.Sequential(alexnet.classifier[0], alexnet.classifier[1], alexnet.classifier[2])
        self.fc7 = nn.Sequential(alexnet.classifier[3], alexnet.classifier[4], alexnet.classifier[5])
        self.fc8 = nn.Linear(4096, num_classes)
        nn.init.normal_(self.fc8.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc8.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc8(self.extract(x, "fc7")["fc7"])

    def extract(self, x: torch.Tensor, layer: str = "fc7") -> OrderedDict[str, torch.Tensor]:
        wanted = _wanted_layers(layer)
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        conv_names = {0: "conv1", 3: "conv2", 6: "conv3", 8: "conv4", 10: "conv5", 12: "pool5"}
        for idx, module in enumerate(self.features):
            x = module(x)
            name = conv_names.get(idx)
            if name in wanted:
                out[name] = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        if "fc6" in wanted:
            out["fc6"] = x
        x = self.fc7(x)
        if "fc7" in wanted:
            out["fc7"] = x
        return out


def _wanted_layers(layer: str) -> set[str]:
    order = ["conv1", "conv2", "conv3", "conv4", "conv5", "pool5", "fc6", "fc7"]
    if layer not in order:
        raise ValueError(f"Unsupported layer {layer}. Choose from {order}.")
    return set(order[: order.index(layer) + 1])


def load_model(path: str | None, num_classes: int = 21, pretrained: bool = True, device: str | torch.device = "cpu") -> RCNNAlexNet:
    model = RCNNAlexNet(num_classes=num_classes, pretrained=pretrained).to(device)
    if path:
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model
