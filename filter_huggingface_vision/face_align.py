"""Five-point face alignment for an ArcFace identity embedding.

Landmarks come from a local RetinaFace (ResNet-34). Weights are a state dict.
Hub repository code is never executed. The five points are warped onto the
ArcFace 112 template with a similarity transform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import nms

from filter_huggingface_vision.utils import resolve_device

logger = logging.getLogger(__name__)

# Canonical ArcFace points on a 112x112 face: left eye, right eye, nose,
# left mouth corner, right mouth corner.
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)
_LANDMARK_REPO = "py-feat/retinaface_r34"
_RGB_MEAN = (123.0, 117.0, 104.0)
_VARIANCE = (0.1, 0.2)
_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_STEPS = (8, 16, 32)
_PRE_NMS_SCORE = 0.02
_NMS_IOU = 0.4
_FACE_SCORE = 0.5
_MATCH_IOU = 0.3


def similarity_matrix(src, dst) -> np.ndarray:
    """2x3 similarity (scale, rotation, translation) mapping ``src`` onto ``dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 2:
        raise ValueError("similarity_matrix expects matching (N, 2) point sets, N >= 2")
    count = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_d = src - mu_src
    dst_d = dst - mu_dst
    cov = (dst_d.T @ src_d) / count
    u_mat, singular, vt_mat = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        sign[-1, -1] = -1
    rotation = u_mat @ sign @ vt_mat
    var_src = float(np.sum(src_d ** 2) / count)
    if var_src <= 0.0:
        raise ValueError("source landmarks are degenerate")
    scale = float(np.trace(np.diag(singular) @ sign) / var_src)
    translation = mu_dst - scale * (rotation @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def warp_face(image: Image.Image, landmarks) -> Image.Image | None:
    """Warp ``image`` so ``landmarks`` land on the ArcFace 112 template."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None
    if np.ptp(points[:, 0]) < 2.0 or np.ptp(points[:, 1]) < 2.0:
        return None
    try:
        matrix = similarity_matrix(points, ARCFACE_112)
    except ValueError:
        return None
    rgb = np.asarray(image.convert("RGB"))
    warped = cv2.warpAffine(
        rgb,
        matrix.astype(np.float32),
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return Image.fromarray(warped)


def _iou(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_landmarks(box, faces, min_iou: float = _MATCH_IOU):
    """Pick the 5 points that belong to ``box``.

    ``faces`` is a list of ``(xyxy, landmarks, score)``. A landmark face whose
    center sits inside the detection wins, highest score first. Otherwise the
    highest IoU wins when it clears ``min_iou``.
    """
    if not faces:
        return None
    inside = []
    best_iou = 0.0
    best_points = None
    for face_box, landmarks, score in faces:
        overlap = _iou(box, face_box)
        if overlap > best_iou:
            best_iou = overlap
            best_points = landmarks
        cx = (float(face_box[0]) + float(face_box[2])) / 2.0
        cy = (float(face_box[1]) + float(face_box[3])) / 2.0
        if float(box[0]) <= cx <= float(box[2]) and float(box[1]) <= cy <= float(box[3]):
            inside.append((float(score), landmarks))
    if inside:
        inside.sort(key=lambda item: item[0], reverse=True)
        return inside[0][1]
    if best_points is not None and best_iou >= min_iou:
        return best_points
    return None


class _ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, slope=None):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if slope is not None:
            layers.append(nn.LeakyReLU(negative_slope=slope, inplace=True))
        super().__init__(*layers)


class _SSH(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.conv3X3 = _ConvBnAct(in_channel, out_channels // 2, 3, slope=None)
        self.conv5X5_1 = _ConvBnAct(in_channel, out_channels // 4, 3, slope=leaky)
        self.conv5X5_2 = _ConvBnAct(out_channels // 4, out_channels // 4, 3, slope=None)
        self.conv7X7_2 = _ConvBnAct(out_channels // 4, out_channels // 4, 3, slope=leaky)
        self.conv7x7_3 = _ConvBnAct(out_channels // 4, out_channels // 4, 3, slope=None)

    def forward(self, x):
        conv3 = self.conv3X3(x)
        conv5 = self.conv5X5_2(self.conv5X5_1(x))
        conv7 = self.conv7x7_3(self.conv7X7_2(self.conv5X5_1(x)))
        return F.relu(torch.cat([conv3, conv5, conv7], dim=1), inplace=True)


class _FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = _ConvBnAct(in_channels[0], out_channels, 1, slope=leaky)
        self.output2 = _ConvBnAct(in_channels[1], out_channels, 1, slope=leaky)
        self.output3 = _ConvBnAct(in_channels[2], out_channels, 1, slope=leaky)
        self.merge1 = _ConvBnAct(out_channels, out_channels, 3, slope=leaky)
        self.merge2 = _ConvBnAct(out_channels, out_channels, 3, slope=leaky)

    def forward(self, features):
        levels = list(features.values())
        output1 = self.output1(levels[0])
        output2 = self.output2(levels[1])
        output3 = self.output3(levels[2])
        output2 = self.merge2(output2 + F.interpolate(output3, size=output2.shape[2:], mode="nearest"))
        output1 = self.merge1(output1 + F.interpolate(output2, size=output1.shape[2:], mode="nearest"))
        return [output1, output2, output3]


class _Head(nn.Module):
    def __init__(self, in_channels, num_anchors, out_per_anchor):
        super().__init__()
        self._out_per_anchor = out_per_anchor
        self._convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, num_anchors * out_per_anchor, kernel_size=1)
                for _ in range(3)
            ]
        )

    def forward(self, features):
        outputs = []
        for feature, layer in zip(features, self._convs):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        return torch.cat(
            [out.view(out.shape[0], -1, self._out_per_anchor) for out in outputs],
            dim=1,
        )


class RetinaFace(nn.Module):
    """RetinaFace ResNet-34. Eval forward returns boxes, scores, and 5 points."""

    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=None)
        self.body = IntermediateLayerGetter(
            backbone, {"layer2": "1", "layer3": "2", "layer4": "3"}
        )
        self.fpn = _FPN([128, 256, 512], 128)
        self.ssh1 = _SSH(128, 128)
        self.ssh2 = _SSH(128, 128)
        self.ssh3 = _SSH(128, 128)
        self.ClassHead = _Head(128, num_anchors=2, out_per_anchor=2)
        self.BboxHead = _Head(128, num_anchors=2, out_per_anchor=4)
        self.LandmarkHead = _Head(128, num_anchors=2, out_per_anchor=10)

    def forward(self, x):
        features = self.fpn(self.body(x))
        heads = [self.ssh1(features[0]), self.ssh2(features[1]), self.ssh3(features[2])]
        boxes = self.BboxHead(heads)
        scores = self.ClassHead(heads)
        landmarks = self.LandmarkHead(heads)
        if self.training:
            return boxes, scores, landmarks
        return boxes, F.softmax(scores, dim=-1), landmarks


def generate_priors(height, width, device):
    anchors = []
    for step, sizes in zip(_STEPS, _MIN_SIZES):
        grid_h = int(np.ceil(height / step))
        grid_w = int(np.ceil(width / step))
        for row in range(grid_h):
            for col in range(grid_w):
                for size in sizes:
                    anchors.append(
                        [
                            (col + 0.5) * step / width,
                            (row + 0.5) * step / height,
                            size / width,
                            size / height,
                        ]
                    )
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def decode_boxes(loc, priors):
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * _VARIANCE[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * _VARIANCE[1]),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre, priors):
    deltas = pre.view(-1, 5, 2)
    cxcy = priors[:, :2].unsqueeze(1)
    wh = priors[:, 2:].unsqueeze(1)
    return (cxcy + deltas * _VARIANCE[0] * wh).view(-1, 10)


class FaceAligner:
    """One landmark forward per frame, then a 112 warp for each detection box."""

    def load(self, device):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weights = hf_hub_download(
            repo_id=_LANDMARK_REPO, filename="model.safetensors", revision="main"
        )
        self._device = resolve_device(device)
        self._model = RetinaFace().to(self._device).eval()
        self._model.load_state_dict(load_file(weights), strict=True)
        self._mean = torch.tensor(_RGB_MEAN, device=self._device).view(1, 3, 1, 1)
        self._priors = {}
        logger.info("Loaded local RetinaFace landmarks from %s", _LANDMARK_REPO)
        return self

    def shutdown(self):
        self._model = None
        self._priors = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prior(self, height, width):
        key = (height, width, str(self._device))
        cached = self._priors.get(key)
        if cached is None:
            cached = generate_priors(height, width, self._device)
            self._priors[key] = cached
        return cached

    @torch.inference_mode()
    def _faces(self, image: Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = rgb.shape[:2]
        batch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        loc, conf, landms = self._model(batch - self._mean)
        priors = self._prior(height, width)
        boxes = decode_boxes(loc[0], priors)
        points = decode_landmarks(landms[0], priors)
        scores = conf[0, :, 1]
        boxes = boxes * torch.tensor([width, height, width, height], device=self._device)
        points = points * torch.tensor([width, height] * 5, device=self._device)
        keep = scores > _PRE_NMS_SCORE
        boxes, points, scores = boxes[keep], points[keep], scores[keep]
        if scores.numel() == 0:
            return []
        if scores.numel() > 5000:
            order = torch.argsort(scores, descending=True)[:5000]
            boxes, points, scores = boxes[order], points[order], scores[order]
        kept = nms(boxes, scores, _NMS_IOU)
        faces = []
        for index in kept.tolist():
            score = float(scores[index])
            if score < _FACE_SCORE:
                continue
            face_box = boxes[index].detach().float().cpu().numpy()
            landmarks = points[index].detach().float().cpu().numpy().reshape(5, 2)
            faces.append((face_box, landmarks, score))
            if len(faces) >= 750:
                break
        return faces

    def align(self, image: Image.Image, boxes):
        if not boxes:
            return []
        faces = self._faces(image)
        aligned = []
        for box in boxes:
            landmarks = match_landmarks(box, faces)
            aligned.append(warp_face(image, landmarks) if landmarks is not None else None)
        return aligned
