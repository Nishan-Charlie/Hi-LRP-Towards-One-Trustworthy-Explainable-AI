"""Image preprocessing and saliency-map utilities.

Preprocessing mirrors the prototype notebook: square-resize to (input_size,
input_size) then ImageNet normalize. Square-resize (rather than
resize-shortest-edge + center-crop) keeps a 1:1 spatial mapping between the
input pixels and any bounding-box / segmentation ground truth, which the
localization metrics rely on.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.transform import resize as sk_resize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image_tensor(path: str, input_size: int = 224) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
    """Return (resized PIL RGB, normalized tensor [1,3,H,W], original (W,H))."""
    img = Image.open(path).convert("RGB")
    orig_wh = img.size  # (W, H)
    tf = build_transform(input_size)
    tensor = tf(img).unsqueeze(0)
    return img.resize((input_size, input_size)), tensor, orig_wh


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization; returns a [3,H,W] tensor in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    x = tensor.detach()
    if x.dim() == 4:
        x = x[0]
    return (x * std + mean).clamp(0, 1)


def to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] normalized tensor -> HxWx3 uint8 array."""
    img = denormalize(tensor).cpu().numpy().transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


def normalize_map(hm: np.ndarray) -> np.ndarray:
    """Min-max normalize a saliency map to [0, 1] (constant maps -> zeros)."""
    hm = np.asarray(hm, dtype=np.float32)
    lo, hi = float(hm.min()), float(hm.max())
    if hi - lo < 1e-12:
        return np.zeros_like(hm)
    return (hm - lo) / (hi - lo)


def resize_map(hm: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a 2D saliency map to (size, size) with anti-aliasing."""
    hm = np.squeeze(np.asarray(hm, dtype=np.float32))
    if hm.shape[0] == size and hm.shape[1] == size:
        return hm
    return sk_resize(hm, (size, size), mode="reflect", anti_aliasing=True).astype(np.float32)
