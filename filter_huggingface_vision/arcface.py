"""Local ArcFace IResNet. Loads Hub weights and never executes repository code."""

import json
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from filter_huggingface_vision.utils import resolve_device

logger = logging.getLogger(__name__)

# InsightFace IResNet depths. ir100 and ir101 are the same [3, 13, 30, 3] net.
_LAYER_COUNTS = {
    "ir18": (2, 2, 2, 2),
    "ir34": (3, 4, 6, 3),
    "ir50": (3, 4, 14, 3),
    "ir100": (3, 13, 30, 3),
    "ir101": (3, 13, 30, 3),
    "ir200": (6, 26, 60, 6),
}
_INPUT_SIZE = 112


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _IBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """ArcFace IResNet. A 112x112 RGB tensor comes back as ``(batch, dim)``."""

    def __init__(self, layers, num_features=512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [_IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(module.bias, 0)
        nn.init.constant_(self.features.weight, 1)
        nn.init.constant_(self.features.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


def build_iresnet(name, num_features=512):
    key = (name or "").lower()
    layers = _LAYER_COUNTS.get(key)
    if layers is None:
        known = ", ".join(sorted(_LAYER_COUNTS))
        raise ValueError(f"unsupported ArcFace architecture {name!r}; expected one of {known}")
    return IResNet(layers, num_features=num_features)


def load_arcface_state(model, state):
    """Load a checkpoint whose keys may be prefixed ``model.net.``."""
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("model.net.", "net."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    leftover = [key for key in missing if not key.endswith("num_batches_tracked")]
    extra = [key for key in unexpected if not key.endswith("num_batches_tracked")]
    if leftover or extra:
        raise ValueError(
            "ArcFace checkpoint does not match the local IResNet "
            f"({len(leftover)} missing, {len(extra)} unexpected keys)"
        )
    return model


class ArcFaceEmbedder:
    """Identity embedder. ``vectors`` returns one ``(dim,)`` row per image."""

    def load(self, config):
        """Load from the same config dict as EmbeddingBackend. Hub code is not run."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model_id = config["model_id"]
        revision = (config.get("revision") or "").strip() or "main"
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", revision=revision)
        with open(config_path, encoding="utf-8") as handle:
            hub_config = json.load(handle)
        conf = hub_config.get("conf") or {}
        name = conf.get("name")
        num_features = int(conf.get("output_dim") or 512)
        weights_path = hf_hub_download(
            repo_id=model_id, filename="model.safetensors", revision=revision
        )
        self._model_id = model_id
        self._device = resolve_device(config.get("device", "cpu"))
        self._model = build_iresnet(name, num_features).to(self._device)
        load_arcface_state(self._model, load_file(weights_path))
        self._model.eval()
        logger.info(
            "Loaded local ArcFace %s %d-d from %s@%s",
            name,
            num_features,
            model_id,
            revision,
        )
        return self

    def shutdown(self):
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _batch(self, images):
        rows = []
        for image in images:
            resized = image.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = (array - 0.5) / 0.5
            rows.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(rows).to(self._device)

    def vectors(self, images):
        if not images:
            return []
        with torch.inference_mode():
            output = self._model(self._batch(images))
        if output.ndim != 2 or output.shape[0] != len(images):
            raise RuntimeError(
                f"ArcFace forward returned {tuple(output.shape)}; expected ({len(images)}, dim)"
            )
        return [row.detach().float().cpu().numpy() for row in output]

    def vector(self, image):
        return self.vectors([image])[0]
