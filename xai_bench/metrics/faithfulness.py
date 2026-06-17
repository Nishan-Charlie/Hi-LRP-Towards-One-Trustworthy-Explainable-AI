import torch
import numpy as np
import quantus

from xai_bench.registry import METRICS

@METRICS.register("deletion_auc")
def deletion_auc(attr: torch.Tensor, model: torch.nn.Module = None, 
                 image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    if model is None or image is None or target is None:
        raise ValueError("Faithfulness metrics require model, image, and target.")
        
    device = image.device
    model.eval()
    
    # Quantus expects batched numpy arrays
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy() # [B, 1, H, W]
    
    metric = quantus.FaithfulnessCorrelation(disable_warnings=True, display_progressbar=False)
    
    try:
        score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: FaithfulnessCorrelation failed: {e}")
        return float('nan')

@METRICS.register("insertion_auc")
def insertion_auc(attr: torch.Tensor, model: torch.nn.Module = None, 
                  image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    if model is None or image is None or target is None:
        raise ValueError("Faithfulness metrics require model, image, and target.")
        
    device = image.device
    model.eval()
    
    x_batch = image.unsqueeze(0).cpu().numpy()
    y_batch = np.array([target])
    a_batch = attr.unsqueeze(0).unsqueeze(0).cpu().numpy()
    
    metric = quantus.FaithfulnessEstimate(disable_warnings=True, display_progressbar=False)
    
    try:
        score = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
        return score[0]
    except Exception as e:
        print(f"Warning: FaithfulnessEstimate failed: {e}")
        return float('nan')
