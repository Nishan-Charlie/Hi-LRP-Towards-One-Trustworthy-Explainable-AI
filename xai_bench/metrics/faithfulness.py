import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc
from copy import deepcopy

from xai_bench.registry import METRICS

def _get_sorted_pixels(attr):
    # attr is [H, W]
    flat_attr = attr.flatten()
    # Sort descending
    sorted_indices = torch.argsort(flat_attr, descending=True)
    return sorted_indices

def _perturb_image(image, indices, step_size, perturb_value=0.0):
    """Perturbs top step_size pixels in the image using indices."""
    # image is [C, H, W]
    C, H, W = image.shape
    perturbed = image.clone()
    flat_img = perturbed.view(C, -1)
    
    # We do cumulative perturbation
    flat_img[:, indices] = perturb_value
    return perturbed.view(C, H, W)

def _compute_auc(model, image, target, attr, mode='deletion', steps=10):
    """Computes AUC for deletion or insertion.
    For deletion, start with orig image, iteratively remove top pixels.
    For insertion, start with blurred/gray image, iteratively add top pixels.
    """
    sorted_indices = _get_sorted_pixels(attr)
    num_pixels = attr.numel()
    step_size = num_pixels // steps
    
    device = image.device
    model.eval()
    
    scores = []
    
    if mode == 'deletion':
        current_img = image.clone()
        baseline_val = 0.0 # gray out
    else: # insertion
        # start from blurred or gray. Let's use gray
        current_img = torch.zeros_like(image)
        baseline_val = image # We insert pixels from original
        
    with torch.no_grad():
        # Step 0: initial state
        logits = model(current_img.unsqueeze(0))
        prob = F.softmax(logits, dim=1)[0, target].item()
        scores.append(prob)
        
        for i in range(1, steps + 1):
            start_idx = (i - 1) * step_size
            end_idx = min(i * step_size, num_pixels)
            chunk_indices = sorted_indices[start_idx:end_idx]
            
            C = current_img.shape[0]
            flat_curr = current_img.view(C, -1)
            
            if mode == 'deletion':
                flat_curr[:, chunk_indices] = baseline_val
            else:
                flat_orig = baseline_val.view(C, -1)
                flat_curr[:, chunk_indices] = flat_orig[:, chunk_indices]
                
            current_img = flat_curr.view_as(current_img)
            
            logits = model(current_img.unsqueeze(0))
            prob = F.softmax(logits, dim=1)[0, target].item()
            scores.append(prob)
            
    # Calculate AUC
    x = np.linspace(0, 1, len(scores))
    return auc(x, scores)

# Note: Faithfulness metrics require the model, image, target, and attr.
# The registry might need a specific signature or wrapper to pass all these.
# We'll assume the runner passes `model, image, target, attr, metadata`.

@METRICS.register("deletion_auc")
def deletion_auc(attr: torch.Tensor, model: torch.nn.Module = None, 
                 image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    if model is None or image is None or target is None:
        raise ValueError("Faithfulness metrics require model, image, and target.")
    return _compute_auc(model, image, target, attr, mode='deletion')

@METRICS.register("insertion_auc")
def insertion_auc(attr: torch.Tensor, model: torch.nn.Module = None, 
                  image: torch.Tensor = None, target: int = None, **kwargs) -> float:
    if model is None or image is None or target is None:
        raise ValueError("Faithfulness metrics require model, image, and target.")
    return _compute_auc(model, image, target, attr, mode='insertion')
