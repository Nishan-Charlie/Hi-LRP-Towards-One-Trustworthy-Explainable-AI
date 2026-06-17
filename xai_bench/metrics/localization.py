import torch
from typing import Dict, List, Any
import numpy as np

from xai_bench.registry import METRICS

def _create_bbox_mask(bboxes, shape):
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.bool)
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        xmin_px = int(max(0, xmin * W))
        ymin_px = int(max(0, ymin * H))
        xmax_px = int(min(W, xmax * W))
        ymax_px = int(min(H, ymax * H))
        mask[ymin_px:ymax_px, xmin_px:xmax_px] = True
    return mask

@METRICS.register("pointing_game")
def pointing_game(attr: torch.Tensor, metadata: Dict[str, Any], **kwargs) -> float:
    """Checks if the maximum attribution pixel falls inside any bounding box."""
    # attr is [H, W]
    if not metadata.get('bboxes'):
        return float('nan')
    mask = _create_bbox_mask(metadata['bboxes'], attr.shape)
    if not mask.any():
        return float('nan')
        
    flat_idx = torch.argmax(attr)
    idx_y, idx_x = flat_idx // attr.shape[1], flat_idx % attr.shape[1]
    
    return 1.0 if mask[idx_y, idx_x] else 0.0

@METRICS.register("ebpg")
def energy_based_pointing_game(attr: torch.Tensor, metadata: Dict[str, Any], **kwargs) -> float:
    """Fraction of positive attribution inside the bounding box."""
    if not metadata.get('bboxes'):
        return float('nan')
    mask = _create_bbox_mask(metadata['bboxes'], attr.shape).to(attr.device)
    if not mask.any():
        return float('nan')
        
    pos_attr = torch.clamp(attr, min=0)
    total_energy = pos_attr.sum()
    if total_energy == 0:
        return 0.0
    inside_energy = (pos_attr * mask).sum()
    return (inside_energy / total_energy).item()

@METRICS.register("iou")
def iou_metric(attr: torch.Tensor, metadata: Dict[str, Any], **kwargs) -> float:
    """IoU of top k% attribution pixels with the bounding box mask."""
    # We binarize attribution using Otsu or a simple top-k%. 
    # Let's use top 20% of image area.
    if not metadata.get('bboxes'):
        return float('nan')
    mask = _create_bbox_mask(metadata['bboxes'], attr.shape).to(attr.device)
    if not mask.any():
        return float('nan')
        
    k = int(attr.numel() * 0.2)
    threshold = torch.kthvalue(attr.flatten(), attr.numel() - k).values
    attr_mask = attr >= threshold
    
    intersection = (attr_mask & mask).float().sum()
    union = (attr_mask | mask).float().sum()
    if union == 0:
        return 0.0
    return (intersection / union).item()
