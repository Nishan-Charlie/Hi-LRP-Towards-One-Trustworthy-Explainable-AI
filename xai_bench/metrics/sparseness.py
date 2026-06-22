import torch
import numpy as np
# pyrefly: ignore [missing-import]
import quantus

from xai_bench.registry import METRICS

def _run_quantus_sparse_metric(metric_class, attr, model, image, target, kwargs, metric_kwargs=None):
    if model is None or image is None or target is None:
        raise ValueError(f"Quantus metric requires model, image, and target.")
        
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    
    device = image.device
    model.eval()
    
    explain_func = kwargs.get('explain_func', None)

    if metric_kwargs is None:
        metric_kwargs = kwargs.get('metric_kwargs', {}) or {}

    metric = metric_class(disable_warnings=True, **metric_kwargs)
    try:
        if explain_func is not None:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device, explain_func=explain_func)
        else:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: {metric_class.__name__} failed: {e}")
        return float('nan')

@METRICS.register("gini")
def gini_coefficient(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_sparse_metric(quantus.Gini, attr, model, image, target, kwargs)

@METRICS.register("entropy")
def shannon_entropy(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_sparse_metric(quantus.Complexity, attr, model, image, target, kwargs)

@METRICS.register("sparseness")
def sparseness(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_sparse_metric(quantus.Sparseness, attr, model, image, target, kwargs)

@METRICS.register("effective_complexity")
def effective_complexity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_sparse_metric(quantus.EffectiveComplexity, attr, model, image, target, kwargs)
