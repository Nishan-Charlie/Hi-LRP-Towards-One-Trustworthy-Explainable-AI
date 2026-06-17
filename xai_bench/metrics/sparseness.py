import torch
import numpy as np
# pyrefly: ignore [missing-import]
import quantus

from xai_bench.registry import METRICS

@METRICS.register("gini")
def gini_coefficient(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    """Computes the Gini coefficient of the attribution map using Quantus.
    Higher Gini = more sparse/concentrated.
    """
    if model is None or image is None or target is None:
        raise ValueError("Quantus Gini requires model, image, and target.")
        
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    
    device = image.device
    model.eval()
    
    metric = quantus.Gini(disable_warnings=True)
    try:
        score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: Gini failed: {e}")
        return float('nan')

@METRICS.register("entropy")
def shannon_entropy(attr: torch.Tensor, model: torch.nn.Module = None, image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    """Computes Complexity (Entropy) of the normalized attribution map using Quantus.
    Lower entropy = more sparse.
    """
    if model is None or image is None or target is None:
        raise ValueError("Quantus Complexity requires model, image, and target.")
        
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    
    device = image.device
    model.eval()
    
    metric = quantus.Complexity(disable_warnings=True)
    try:
        score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: Complexity failed: {e}")
        return float('nan')
