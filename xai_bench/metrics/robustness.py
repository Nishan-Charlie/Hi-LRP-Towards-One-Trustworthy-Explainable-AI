import torch
import numpy as np
import quantus

from xai_bench.registry import METRICS

def _run_quantus_robustness_metric(metric_class, attr, model, image, target, kwargs, metric_kwargs=None):
    if model is None or image is None or target is None:
        raise ValueError(f"Quantus robustness metric requires model, image, and target.")
        
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

@METRICS.register("continuity")
def continuity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.Continuity, attr, model, image, target, kwargs)

@METRICS.register("local_lipschitz_estimate")
def local_lipschitz_estimate(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.LocalLipschitzEstimate, attr, model, image, target, kwargs)

@METRICS.register("max_sensitivity")
def max_sensitivity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.MaxSensitivity, attr, model, image, target, kwargs)

@METRICS.register("avg_sensitivity")
def avg_sensitivity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.AvgSensitivity, attr, model, image, target, kwargs)

@METRICS.register("consistency")
def consistency(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.Consistency, attr, model, image, target, kwargs)

@METRICS.register("relative_input_stability")
def relative_input_stability(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.RelativeInputStability, attr, model, image, target, kwargs)

@METRICS.register("relative_output_stability")
def relative_output_stability(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.RelativeOutputStability, attr, model, image, target, kwargs)

@METRICS.register("relative_representation_stability")
def relative_representation_stability(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_robustness_metric(quantus.RelativeRepresentationStability, attr, model, image, target, kwargs)
