import torch
from typing import Dict, List, Any
import numpy as np
import quantus

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

def _run_quantus_loc_metric(metric_class, attr, metadata, model, image, target, kwargs, metric_kwargs=None):
    if not metadata.get('bboxes'):
        return float('nan')
    if model is None or image is None or target is None:
        raise ValueError(f"Quantus metric requires model, image, and target.")
        
    mask = _create_bbox_mask(metadata['bboxes'], attr.shape)
    if not mask.any():
        return float('nan')
        
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    s_batch = mask.unsqueeze(0).unsqueeze(0).cpu().numpy().astype(float)
    
    device = image.device
    model.eval()
    
    explain_func = kwargs.get('explain_func', None)

    if metric_kwargs is None:
        metric_kwargs = kwargs.get('metric_kwargs', {}) or {}

    metric = metric_class(disable_warnings=True, **metric_kwargs)
    try:
        if explain_func is not None:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, s_batch=s_batch, device=device, explain_func=explain_func)
        else:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, s_batch=s_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: {metric_class.__name__} failed: {e}")
        return float('nan')

@METRICS.register("pointing_game")
def pointing_game(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.PointingGame, attr, metadata, model, image, target, kwargs)

@METRICS.register("top_k_intersection")
def top_k_intersection(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.TopKIntersection, attr, metadata, model, image, target, kwargs)

@METRICS.register("relevance_mass_accuracy")
def relevance_mass_accuracy(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.RelevanceMassAccuracy, attr, metadata, model, image, target, kwargs)

@METRICS.register("relevance_rank_accuracy")
def relevance_rank_accuracy(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.RelevanceRankAccuracy, attr, metadata, model, image, target, kwargs)

@METRICS.register("attribution_localisation")
def attribution_localisation(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.AttributionLocalisation, attr, metadata, model, image, target, kwargs)

@METRICS.register("auc")
def auc_loc(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.AUC, attr, metadata, model, image, target, kwargs)

@METRICS.register("focus")
def focus(attr: torch.Tensor, metadata: Dict[str, Any], model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_loc_metric(quantus.Focus, attr, metadata, model, image, target, kwargs)
