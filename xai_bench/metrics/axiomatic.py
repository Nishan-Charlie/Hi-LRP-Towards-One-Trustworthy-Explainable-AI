import torch
import numpy as np
import quantus

from xai_bench.registry import METRICS

def _run_quantus_axiomatic_metric(metric_class, attr, model, image, target, kwargs, metric_kwargs=None):
    if model is None or image is None or target is None:
        raise ValueError(f"Quantus axiomatic metric requires model, image, and target.")
        
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    
    device = image.device
    model.eval()
    
    explain_func = kwargs.get('explain_func', None)
    
    if metric_kwargs is None:
        metric_kwargs = {}
    
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

@METRICS.register("completeness")
def completeness(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_axiomatic_metric(quantus.Completeness, attr, model, image, target, kwargs)

@METRICS.register("non_sensitivity")
def non_sensitivity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_axiomatic_metric(quantus.NonSensitivity, attr, model, image, target, kwargs)

@METRICS.register("input_invariance")
def input_invariance(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_axiomatic_metric(quantus.InputInvariance, attr, model, image, target, kwargs)
