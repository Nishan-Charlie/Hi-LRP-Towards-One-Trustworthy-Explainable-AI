import torch
import numpy as np
import quantus

from xai_bench.registry import METRICS

def _run_quantus_metric(metric_class, attr, model, image, target, kwargs, metric_kwargs=None):
    if model is None or image is None or target is None:
        raise ValueError(f"Quantus metric requires model, image, and target.")
        
    device = image.device
    model.eval()
    
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy() # [B, 1, H, W]
    
    explain_func = kwargs.get('explain_func', None)

    if metric_kwargs is None:
        metric_kwargs = kwargs.get('metric_kwargs', {}) or {}

    metric = metric_class(disable_warnings=True, display_progressbar=False, **metric_kwargs)
    
    try:
        # If the metric needs an explain_func, pass it if available
        if explain_func is not None:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device, explain_func=explain_func)
        else:
            score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: {metric_class.__name__} failed: {e}")
        return float('nan')

def apf_perturb_func(arr: np.ndarray, indices: tuple, **kwargs) -> np.ndarray:
    """
    Attention-Preserving Faithfulness (APF) perturbation function.
    Instead of replacing with a static baseline, it replaces masked pixels with the
    mean of the unmasked regions to preserve sequence statistics.
    """
    arr_perturbed = np.copy(arr)
    
    if indices is None:
        return arr_perturbed
    if isinstance(indices, (tuple, list)) and (len(indices) == 0 or len(indices[0]) == 0):
        return arr_perturbed
    if isinstance(indices, np.ndarray) and indices.size == 0:
        return arr_perturbed

    mask = np.zeros(arr.shape[1:], dtype=bool)
    mask[indices] = True
    
    for c in range(arr.shape[0]):
        unmasked = arr[c][~mask]
        mean_val = float(np.mean(unmasked)) if unmasked.size > 0 else 0.0
        arr_perturbed[c, mask] = mean_val
        
    return arr_perturbed

@METRICS.register("apf")
def attention_preserving_faithfulness(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    # Use FaithfulnessCorrelation but with our APF perturbation function
    kwargs.setdefault('metric_kwargs', {})
    kwargs['metric_kwargs']['perturb_func'] = apf_perturb_func
    return _run_quantus_metric(quantus.FaithfulnessCorrelation, attr, model, image, target, kwargs)

@METRICS.register("faithfulness_correlation")
def faithfulness_correlation(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.FaithfulnessCorrelation, attr, model, image, target, kwargs)

@METRICS.register("faithfulness_estimate")
def faithfulness_estimate(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.FaithfulnessEstimate, attr, model, image, target, kwargs)

@METRICS.register("pixel_flipping")
def pixel_flipping(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.PixelFlipping, attr, model, image, target, kwargs)

@METRICS.register("region_perturbation")
def region_perturbation(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.RegionPerturbation, attr, model, image, target, kwargs)

@METRICS.register("monotonicity_arya")
def monotonicity_arya(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.MonotonicityCorrelation, attr, model, image, target, kwargs)

@METRICS.register("monotonicity_nguyen")
def monotonicity_nguyen(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.Monotonicity, attr, model, image, target, kwargs)

@METRICS.register("selectivity")
def selectivity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.Selectivity, attr, model, image, target, kwargs)

@METRICS.register("sensitivity_n")
def sensitivity_n(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.SensitivityN, attr, model, image, target, kwargs)

@METRICS.register("irof")
def irof(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.IROF, attr, model, image, target, kwargs)

@METRICS.register("road")
def road(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.ROAD, attr, model, image, target, kwargs)

@METRICS.register("infidelity")
def infidelity(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.Infidelity, attr, model, image, target, kwargs)

@METRICS.register("sufficiency")
def sufficiency(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    return _run_quantus_metric(quantus.Sufficiency, attr, model, image, target, kwargs)
